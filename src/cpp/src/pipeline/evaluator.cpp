//
// Created by Jason Mohoney on 2/28/20.
//

#include "pipeline/evaluator.h"

#include "configuration/constants.h"
#include "reporting/logger.h"
#include <chrono>

PipelineEvaluator::PipelineEvaluator(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model, shared_ptr<PipelineConfig> pipeline_config) {
    dataloader_ = dataloader;

    if (model->device_.is_cuda()) {
        pipeline_ = std::make_shared<PipelineGPU>(dataloader, model, false, nullptr, pipeline_config);
    } else {
        pipeline_ = std::make_shared<PipelineCPU>(dataloader, model, false, nullptr, pipeline_config);
    }

    pipeline_->initialize();
}

void PipelineEvaluator::evaluate(bool validation) {
    auto total_eval_start = std::chrono::high_resolution_clock::now();
    
    if (!dataloader_->single_dataset_) {
        auto dataset_switch_start = std::chrono::high_resolution_clock::now();
        if (validation) {
            SPDLOG_INFO("Evaluating validation set");
            dataloader_->setValidationSet();
        } else {
            SPDLOG_INFO("Evaluating test set");
            dataloader_->setTestSet();
        }
        auto dataset_switch_end = std::chrono::high_resolution_clock::now();
        double dataset_switch_time = std::chrono::duration<double, std::milli>(dataset_switch_end - dataset_switch_start).count();
        SPDLOG_INFO("  Dataset switch: {:.2f}ms", dataset_switch_time);
    }

    auto batch_init_start = std::chrono::high_resolution_clock::now();
    dataloader_->initializeBatches(false);
    auto batch_init_end = std::chrono::high_resolution_clock::now();
    double batch_init_time = std::chrono::duration<double, std::milli>(batch_init_end - batch_init_start).count();
    SPDLOG_INFO("  Evaluation batch initialization: {:.2f}ms", batch_init_time);

    if (dataloader_->evaluation_negative_sampler_ != nullptr) {
        if (dataloader_->evaluation_config_->negative_sampling->filtered) {
            auto sort_start = std::chrono::high_resolution_clock::now();
            dataloader_->graph_storage_->sortAllEdges();
            auto sort_end = std::chrono::high_resolution_clock::now();
            double sort_time = std::chrono::duration<double, std::milli>(sort_end - sort_start).count();
            SPDLOG_INFO("  Edge sorting: {:.2f}ms", sort_time);
        }
    }

    Timer timer = Timer(false);
    timer.start();
    
    auto pipeline_start_time = std::chrono::high_resolution_clock::now();
    pipeline_->start();
    auto pipeline_start_end = std::chrono::high_resolution_clock::now();
    double pipeline_start_duration = std::chrono::duration<double, std::milli>(pipeline_start_end - pipeline_start_time).count();
    SPDLOG_INFO("  Evaluation pipeline start: {:.2f}ms", pipeline_start_duration);
    
    auto computation_start = std::chrono::high_resolution_clock::now();
    pipeline_->waitComplete();
    auto computation_end = std::chrono::high_resolution_clock::now();
    double computation_time = std::chrono::duration<double, std::milli>(computation_end - computation_start).count();
    SPDLOG_INFO("  Evaluation computation: {:.2f}ms", computation_time);
    
    auto flush_start = std::chrono::high_resolution_clock::now();
    pipeline_->pauseAndFlush();
    auto flush_end = std::chrono::high_resolution_clock::now();
    double flush_time = std::chrono::duration<double, std::milli>(flush_end - flush_start).count();
    SPDLOG_INFO("  Pipeline flush: {:.2f}ms", flush_time);
    
    auto report_start = std::chrono::high_resolution_clock::now();
    pipeline_->model_->reporter_->report();
    auto report_end = std::chrono::high_resolution_clock::now();
    double report_time = std::chrono::duration<double, std::milli>(report_end - report_start).count();
    SPDLOG_INFO("  Results reporting: {:.2f}ms", report_time);
    
    timer.stop();

    auto total_eval_end = std::chrono::high_resolution_clock::now();
    double total_eval_time = std::chrono::duration<double, std::milli>(total_eval_end - total_eval_start).count();

    int64_t epoch_time = timer.getDuration();
    SPDLOG_INFO("Evaluation complete: {}ms (detailed total: {:.2f}ms)", epoch_time, total_eval_time);
}

SynchronousEvaluator::SynchronousEvaluator(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model) {
    dataloader_ = dataloader;
    model_ = model;
}

void SynchronousEvaluator::evaluate(bool validation) {
    auto total_eval_start = std::chrono::high_resolution_clock::now();
    
    if (!dataloader_->single_dataset_) {
        auto dataset_switch_start = std::chrono::high_resolution_clock::now();
        if (validation) {
            SPDLOG_INFO("Evaluating validation set");
            dataloader_->setValidationSet();
        } else {
            SPDLOG_INFO("Evaluating test set");
            dataloader_->setTestSet();
        }
        auto dataset_switch_end = std::chrono::high_resolution_clock::now();
        double dataset_switch_time = std::chrono::duration<double, std::milli>(dataset_switch_end - dataset_switch_start).count();
        SPDLOG_INFO("  Dataset switch: {:.2f}ms", dataset_switch_time);
    }

    auto batch_init_start = std::chrono::high_resolution_clock::now();
    dataloader_->initializeBatches(false);
    auto batch_init_end = std::chrono::high_resolution_clock::now();
    double batch_init_time = std::chrono::duration<double, std::milli>(batch_init_end - batch_init_start).count();
    SPDLOG_INFO("  Evaluation batch initialization: {:.2f}ms", batch_init_time);

    if (dataloader_->evaluation_negative_sampler_ != nullptr) {
        if (dataloader_->evaluation_config_->negative_sampling->filtered) {
            auto sort_start = std::chrono::high_resolution_clock::now();
            dataloader_->graph_storage_->sortAllEdges();
            auto sort_end = std::chrono::high_resolution_clock::now();
            double sort_time = std::chrono::duration<double, std::milli>(sort_end - sort_start).count();
            SPDLOG_INFO("  Edge sorting: {:.2f}ms", sort_time);
        }
    }

    Timer timer = Timer(false);
    timer.start();
    int num_batches = 0;

    auto batch_processing_start = std::chrono::high_resolution_clock::now();
    double total_batch_to_device_time = 0.0;
    double total_gpu_load_time = 0.0;
    double total_model_eval_time = 0.0;
    
    while (dataloader_->hasNextBatch()) {
        shared_ptr<Batch> batch = dataloader_->getBatch();
        
        auto batch_to_device_start = std::chrono::high_resolution_clock::now();
        if (dataloader_->graph_storage_->embeddingsOffDevice()) {
            batch->to(model_->device_);
        }
        auto batch_to_device_end = std::chrono::high_resolution_clock::now();
        total_batch_to_device_time += std::chrono::duration<double, std::milli>(batch_to_device_end - batch_to_device_start).count();
        
        auto gpu_load_start = std::chrono::high_resolution_clock::now();
        dataloader_->loadGPUParameters(batch);
        auto gpu_load_end = std::chrono::high_resolution_clock::now();
        total_gpu_load_time += std::chrono::duration<double, std::milli>(gpu_load_end - gpu_load_start).count();

        auto model_eval_start = std::chrono::high_resolution_clock::now();
        model_->evaluate_batch(batch);
        auto model_eval_end = std::chrono::high_resolution_clock::now();
        total_model_eval_time += std::chrono::duration<double, std::milli>(model_eval_end - model_eval_start).count();

        dataloader_->finishedBatch();
        batch->clear();
        num_batches++;
    }
    auto batch_processing_end = std::chrono::high_resolution_clock::now();
    double batch_processing_time = std::chrono::duration<double, std::milli>(batch_processing_end - batch_processing_start).count();
    
    timer.stop();

    SPDLOG_INFO("  Batch processing breakdown:");
    SPDLOG_INFO("    Total batches: {}", num_batches);
    SPDLOG_INFO("    Batch to device: {:.2f}ms total, {:.2f}ms avg", total_batch_to_device_time, total_batch_to_device_time / num_batches);
    SPDLOG_INFO("    GPU parameter loading: {:.2f}ms total, {:.2f}ms avg", total_gpu_load_time, total_gpu_load_time / num_batches);
    SPDLOG_INFO("    Model evaluation: {:.2f}ms total, {:.2f}ms avg", total_model_eval_time, total_model_eval_time / num_batches);
    SPDLOG_INFO("    Total batch processing: {:.2f}ms", batch_processing_time);

    auto report_start = std::chrono::high_resolution_clock::now();
    model_->reporter_->report();
    auto report_end = std::chrono::high_resolution_clock::now();
    double report_time = std::chrono::duration<double, std::milli>(report_end - report_start).count();
    SPDLOG_INFO("  Results reporting: {:.2f}ms", report_time);
    
    auto total_eval_end = std::chrono::high_resolution_clock::now();
    double total_eval_time = std::chrono::duration<double, std::milli>(total_eval_end - total_eval_start).count();

    int64_t epoch_time = timer.getDuration();
    SPDLOG_INFO("Evaluation complete: {}ms (detailed total: {:.2f}ms)", epoch_time, total_eval_time);
}