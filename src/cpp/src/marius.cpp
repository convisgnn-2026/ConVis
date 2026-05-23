
#include "marius.h"

#include "common/util.h"
#include "configuration/util.h"
#include "pipeline/evaluator.h"
#include "pipeline/graph_encoder.h"
#include "pipeline/trainer.h"
#include "reporting/logger.h"
#include "storage/checkpointer.h"
#include "storage/io.h"
#include <chrono>

static double total_trainer_time = 0.0;  // NEW: Track total trainer time separately

void encode_and_export(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model, shared_ptr<MariusConfig> marius_config) {
    shared_ptr<GraphEncoder> graph_encoder;
    if (marius_config->evaluation->pipeline->sync) {
        graph_encoder = std::make_shared<SynchronousGraphEncoder>(dataloader, model);
    } else {
        graph_encoder = std::make_shared<PipelineGraphEncoder>(dataloader, model, marius_config->evaluation->pipeline);
    }

    string filename = marius_config->storage->model_dir + PathConstants::encoded_nodes_file + PathConstants::file_ext;

    if (fileExists(filename)) {
        remove(filename.c_str());
    }

    int64_t num_nodes = marius_config->storage->dataset->num_nodes;

    int last_stage = marius_config->model->encoder->layers.size() - 1;
    int last_layer = marius_config->model->encoder->layers[last_stage].size() - 1;
    int64_t dim = marius_config->model->encoder->layers[last_stage][last_layer]->output_dim;

    dataloader->graph_storage_->storage_ptrs_.encoded_nodes = std::make_shared<FlatFile>(filename, num_nodes, dim, torch::kFloat32, true);

    graph_encoder->encode();
}

std::tuple<shared_ptr<Model>, shared_ptr<GraphModelStorage>, shared_ptr<DataLoader> > marius_init(shared_ptr<MariusConfig> marius_config, bool train) {
    Timer initialization_timer = Timer(false);
    initialization_timer.start();
    SPDLOG_INFO("Start initialization");

    MariusLogger marius_logger = MariusLogger(marius_config->storage->model_dir);
    spdlog::set_default_logger(marius_logger.main_logger_);
    marius_logger.setConsoleLogLevel(marius_config->storage->log_level);

    torch::manual_seed(marius_config->model->random_seed);
    srand(marius_config->model->random_seed);

    std::vector<torch::Device> devices = devices_from_config(marius_config->storage);

    shared_ptr<Model> model;
    shared_ptr<GraphModelStorage> graph_model_storage;

    int epochs_processed = 0;

    if (train) {
        // initialize new model
        if (!marius_config->training->resume_training && marius_config->training->resume_from_checkpoint.empty()) {
            model = initModelFromConfig(marius_config->model, devices, marius_config->storage->dataset->num_relations, true);
            graph_model_storage = initializeStorage(model, marius_config->storage, !marius_config->training->resume_training, true);
        } else {
            auto checkpoint_loader = std::make_shared<Checkpointer>();

            string checkpoint_dir = marius_config->storage->model_dir;
            if (!marius_config->training->resume_from_checkpoint.empty()) {
                checkpoint_dir = marius_config->training->resume_from_checkpoint;
            }

            auto tup = checkpoint_loader->load(checkpoint_dir, marius_config, true);
            model = std::get<0>(tup);
            graph_model_storage = std::get<1>(tup);

            CheckpointMeta checkpoint_meta = std::get<2>(tup);
            epochs_processed = checkpoint_meta.num_epochs;
        }
    } else {
        auto checkpoint_loader = std::make_shared<Checkpointer>();

        string checkpoint_dir = marius_config->storage->model_dir;
        if (!marius_config->evaluation->checkpoint_dir.empty()) {
            checkpoint_dir = marius_config->evaluation->checkpoint_dir;
        }
        auto tup = checkpoint_loader->load(checkpoint_dir, marius_config, false);
        model = std::get<0>(tup);
        graph_model_storage = std::get<1>(tup);

        CheckpointMeta checkpoint_meta = std::get<2>(tup);
        epochs_processed = checkpoint_meta.num_epochs;
    }

    shared_ptr<DataLoader> dataloader = std::make_shared<DataLoader>(graph_model_storage, model->learning_task_, marius_config->training,
                                                                     marius_config->evaluation, marius_config->model->encoder);

    dataloader->epochs_processed_ = epochs_processed;

    initialization_timer.stop();
    int64_t initialization_time = initialization_timer.getDuration();

    SPDLOG_INFO("Initialization Complete: {}s", (double)initialization_time / 1000);

    return std::forward_as_tuple(model, graph_model_storage, dataloader);
}

void marius_train(shared_ptr<MariusConfig> marius_config) {
    total_trainer_time = 0.0;  // NEW: Reset this too

    // Start total training time measurement
    Timer total_training_timer = Timer(false);
    total_training_timer.start();

    SPDLOG_INFO("=== TRAINING SETUP BREAKDOWN ===");
    
    auto init_start = std::chrono::high_resolution_clock::now();
    auto tup = marius_init(marius_config, true);
    auto init_end = std::chrono::high_resolution_clock::now();
    double init_time = std::chrono::duration<double, std::milli>(init_end - init_start).count();
    SPDLOG_INFO("marius_init(): {:.2f}ms", init_time);
    
    auto model_extract_start = std::chrono::high_resolution_clock::now();
    auto model = std::get<0>(tup);
    auto graph_model_storage = std::get<1>(tup);
    auto dataloader = std::get<2>(tup);
    auto model_extract_end = std::chrono::high_resolution_clock::now();
    double model_extract_time = std::chrono::duration<double, std::milli>(model_extract_end - model_extract_start).count();
    SPDLOG_INFO("Component extraction: {:.2f}ms", model_extract_time);

    shared_ptr<Trainer> trainer;
    shared_ptr<Evaluator> evaluator;

    auto checkpoint_start = std::chrono::high_resolution_clock::now();
    shared_ptr<Checkpointer> model_saver;
    CheckpointMeta metadata;
    if (marius_config->training->save_model) {
        model_saver = std::make_shared<Checkpointer>(model, graph_model_storage, marius_config->training->checkpoint);
        metadata.has_state = true;
        metadata.has_encoded = marius_config->storage->export_encoded_nodes;
        metadata.has_model = true;
        metadata.link_prediction = marius_config->model->learning_task == LearningTask::LINK_PREDICTION;
    }
    auto checkpoint_end = std::chrono::high_resolution_clock::now();
    double checkpoint_time = std::chrono::duration<double, std::milli>(checkpoint_end - checkpoint_start).count();
    SPDLOG_INFO("Checkpointer setup: {:.2f}ms", checkpoint_time);

    auto trainer_create_start = std::chrono::high_resolution_clock::now();
    if (marius_config->training->pipeline->sync) {
        trainer = std::make_shared<SynchronousTrainer>(dataloader, model, marius_config->training->logs_per_epoch);
    } else {
        trainer = std::make_shared<PipelineTrainer>(dataloader, model, marius_config->training->pipeline, marius_config->training->logs_per_epoch);
    }
    auto trainer_create_end = std::chrono::high_resolution_clock::now();
    double trainer_create_time = std::chrono::duration<double, std::milli>(trainer_create_end - trainer_create_start).count();
    SPDLOG_INFO("Trainer creation: {:.2f}ms", trainer_create_time);

    auto evaluator_create_start = std::chrono::high_resolution_clock::now();
    if (marius_config->evaluation->pipeline->sync) {
        evaluator = std::make_shared<SynchronousEvaluator>(dataloader, model);
    } else {
        evaluator = std::make_shared<PipelineEvaluator>(dataloader, model, marius_config->evaluation->pipeline);
    }
    auto evaluator_create_end = std::chrono::high_resolution_clock::now();
    double evaluator_create_time = std::chrono::duration<double, std::milli>(evaluator_create_end - evaluator_create_start).count();
    SPDLOG_INFO("Evaluator creation: {:.2f}ms", evaluator_create_time);

    auto setup_finalization_start = std::chrono::high_resolution_clock::now();
    int checkpoint_interval = marius_config->training->checkpoint->interval;
    auto setup_finalization_end = std::chrono::high_resolution_clock::now();
    double setup_finalization_time = std::chrono::duration<double, std::milli>(setup_finalization_end - setup_finalization_start).count();
    
    double total_setup_time = init_time + model_extract_time + checkpoint_time + trainer_create_time + evaluator_create_time + setup_finalization_time;
    SPDLOG_INFO("Setup finalization: {:.2f}ms", setup_finalization_time);
    SPDLOG_INFO("=== TOTAL TRAINING SETUP: {:.2f}ms ===", total_setup_time);
    
    for (int epoch = 0; epoch < marius_config->training->num_epochs; epoch++) {
        // === TRAINING (measure this specifically) ===
        if (epoch == 0) {
            SPDLOG_INFO("=== FIRST EPOCH TRAINING START ===");
            auto first_train_start = std::chrono::high_resolution_clock::now();
            Timer trainer_timer = Timer(false);
            trainer_timer.start();
            trainer->train(1);
            trainer_timer.stop();
            auto first_train_end = std::chrono::high_resolution_clock::now();
            double first_train_detailed = std::chrono::duration<double, std::milli>(first_train_end - first_train_start).count();
            double epoch_trainer_time = (double)trainer_timer.getDuration();
            total_trainer_time += epoch_trainer_time;
            SPDLOG_INFO("First epoch training (detailed): {:.2f}ms", first_train_detailed);
            SPDLOG_INFO("First epoch training (timer): {}ms", epoch_trainer_time);
        } else {
            Timer trainer_timer = Timer(false);
            trainer_timer.start();
            trainer->train(1);
            trainer_timer.stop();
            double epoch_trainer_time = (double)trainer_timer.getDuration();
            total_trainer_time += epoch_trainer_time;
        }

        if ((epoch + 1) % marius_config->evaluation->epochs_per_eval == 0) {
            if (marius_config->storage->dataset->num_valid != -1) {
                evaluator->evaluate(true);
            }

            if (marius_config->storage->dataset->num_test != -1) {
                evaluator->evaluate(false);
            }
        }

        metadata.num_epochs = dataloader->epochs_processed_;
        if (checkpoint_interval > 0 && (epoch + 1) % checkpoint_interval == 0 && epoch + 1 < marius_config->training->num_epochs) {
            model_saver->create_checkpoint(marius_config->storage->model_dir, metadata, dataloader->epochs_processed_);
        }
    }

    if (marius_config->training->save_model) {
        model_saver->save(marius_config->storage->model_dir, metadata);

        if (marius_config->storage->export_encoded_nodes) {
            encode_and_export(dataloader, model, marius_config);
        }
    }

    total_training_timer.stop();
    double total_time_ms = (double)total_training_timer.getDuration();
    SPDLOG_INFO("");
    SPDLOG_INFO("--- Total Training Time ---");
    SPDLOG_INFO("Total time sec:            {:.6f} seconds", 
               total_time_ms / 1000.0);
}

void marius_eval(shared_ptr<MariusConfig> marius_config) {
    auto tup = marius_init(marius_config, false);
    auto model = std::get<0>(tup);
    auto graph_model_storage = std::get<1>(tup);
    auto dataloader = std::get<2>(tup);

    shared_ptr<Evaluator> evaluator;

    if (marius_config->evaluation->epochs_per_eval > 0) {
        if (marius_config->evaluation->pipeline->sync) {
            evaluator = std::make_shared<SynchronousEvaluator>(dataloader, model);
        } else {
            evaluator = std::make_shared<PipelineEvaluator>(dataloader, model, marius_config->evaluation->pipeline);
        }
        evaluator->evaluate(false);
    }

    if (marius_config->storage->export_encoded_nodes) {
        encode_and_export(dataloader, model, marius_config);
    }
}

void marius(int argc, char *argv[]) {
    (void)argc;

    bool train = true;
    string command_path = string(argv[0]);
    string config_path = string(argv[1]);
    string command_name = command_path.substr(command_path.find_last_of("/\\") + 1);
    if (strcmp(command_name.c_str(), "marius_eval") == 0) {
        train = false;
    }

    shared_ptr<MariusConfig> marius_config = loadConfig(config_path, true);

    if (train) {
        marius_train(marius_config);
    } else {
        marius_eval(marius_config);
    }
}

int main(int argc, char *argv[]) { marius(argc, argv); }