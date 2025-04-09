//
// Created by ssunah on 12/3/18.
//

#include <chrono>
#include <future>
#include <thread>
#include <fstream>
#include <vector>
#include <sstream>
#include <functional>
#include <filesystem>
#include <dirent.h>
#include "matchingcommand.h"
#include "graph/graph.h"
#include "GenerateFilteringPlan.h"
#include "FilterVertices.h"
#include "BuildTable.h"
#include "GenerateQueryPlan.h"
#include "EvaluateQuery.h"

#define NANOSECTOSEC(elapsed_time) ((elapsed_time) / (double)1000000000)
#define BYTESTOMB(memory_cost) ((memory_cost) / (double)(1024 * 1024))

size_t enumerate(Graph *data_graph, Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
                 ui *matching_order, size_t output_limit)
{
    static ui order_id = 0;

    order_id += 1;

    auto start = std::chrono::high_resolution_clock::now();
    size_t call_count = 0;
    size_t embedding_count = EvaluateQuery::LFTJ(data_graph, query_graph, edge_matrix, candidates, candidates_count,
                                                 matching_order, output_limit, call_count);

    auto end = std::chrono::high_resolution_clock::now();
    double enumeration_time_in_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
#ifdef SPECTRUM
    if (EvaluateQuery::exit_)
    {
        printf("Spectrum Order %u status: Timeout\n", order_id);
    }
    else
    {
        printf("Spectrum Order %u status: Complete\n", order_id);
    }
#endif
    printf("Spectrum Order %u Enumerate time (seconds): %.4lf\n", order_id, NANOSECTOSEC(enumeration_time_in_ns));
    printf("Spectrum Order %u #Embeddings: %zu\n", order_id, embedding_count);
    printf("Spectrum Order %u Call Count: %zu\n", order_id, call_count);
    printf("Spectrum Order %u Per Call Count Time (nanoseconds): %.4lf\n", order_id, enumeration_time_in_ns / (call_count == 0 ? 1 : call_count));

    return embedding_count;
}

std::string get_last_name(const std::string &path)
{
    size_t last_sep = path.find_last_of("/");
    if (last_sep != std::string::npos)
    {
        return path.substr(last_sep + 1);
    }
    return path;
}

void spectrum_analysis(Graph *data_graph, Graph *query_graph, Edges ***edge_matrix, ui **candidates, ui *candidates_count,
                       size_t output_limit, std::vector<std::vector<ui>> &spectrum, size_t time_limit_in_sec)
{

    for (auto &order : spectrum)
    {
        std::cout << "----------------------------" << std::endl;
        ui *matching_order = order.data();
        GenerateQueryPlan::printSimplifiedQueryPlan(query_graph, matching_order);

        std::future<size_t> future = std::async(std::launch::async, [data_graph, query_graph, edge_matrix, candidates, candidates_count,
                                                                     matching_order, output_limit]()
                                                { return enumerate(data_graph, query_graph, edge_matrix, candidates, candidates_count, matching_order, output_limit); });

        std::cout << "execute...\n";
        std::future_status status;
        do
        {
            status = future.wait_for(std::chrono::seconds(time_limit_in_sec));
            if (status == std::future_status::deferred)
            {
                std::cout << "Deferred\n";
                exit(-1);
            }
            else if (status == std::future_status::timeout)
            {
#ifdef SPECTRUM
                EvaluateQuery::exit_ = true;
#endif
            }
        } while (status != std::future_status::ready);
    }
}

int main(int argc, char **argv)
{
    MatchingCommand command(argc, argv);
    std::string input_query_graph_file = command.getQueryGraphFilePath();
    std::string input_data_graph_file = command.getDataGraphFilePath();
    std::string input_filter_type = command.getFilterType();
    std::string input_order_type = command.getOrderType();
    std::string input_engine_type = command.getEngineType();
    std::string input_max_embedding_num = command.getMaximumEmbeddingNum();
    std::string input_time_limit = command.getTimeLimit();
    std::string input_order_num = command.getOrderNum();
    std::string input_distribution_file_path = command.getDistributionFilePath();
    std::string input_csr_file_path = command.getCSRFilePath();
    std::string input_mode = command.getMode();

    Graph *data_graph = new Graph(true);

    if (input_csr_file_path.empty())
    {
        data_graph->loadGraphFromFile(input_data_graph_file);
    }
    else
    {
        std::string degree_file_path = input_csr_file_path + "_deg.bin";
        std::string edge_file_path = input_csr_file_path + "_adj.bin";
        std::string label_file_path = input_csr_file_path + "_label.bin";
        data_graph->loadGraphFromFileCompressed(degree_file_path, edge_file_path, label_file_path);
    }

    Graph *query_graph = new Graph(true);
    query_graph->loadGraphFromFile(input_query_graph_file);
    query_graph->buildCoreTable();

    auto start = std::chrono::high_resolution_clock::now();

    ui **candidates = NULL;
    ui *candidates_count = NULL;
    std::vector<std::unordered_map<VertexID, std::vector<VertexID>>> TE_Candidates;
    std::vector<std::vector<std::unordered_map<VertexID, std::vector<VertexID>>>> NTE_Candidates;

    // For GQL Filter
    FilterVertices::GQLFilter(data_graph, query_graph, candidates, candidates_count);
    FilterVertices::sortCandidates(candidates, candidates_count, query_graph->getVerticesCount());

    auto end = std::chrono::high_resolution_clock::now();
    double filter_vertices_time_in_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    start = std::chrono::high_resolution_clock::now();

    Edges ***edge_matrix = NULL;

    edge_matrix = new Edges **[query_graph->getVerticesCount()];
    for (ui i = 0; i < query_graph->getVerticesCount(); ++i)
    {
        edge_matrix[i] = new Edges *[query_graph->getVerticesCount()];
    }

    BuildTable::buildTables(data_graph, query_graph, candidates, candidates_count, edge_matrix);

    end = std::chrono::high_resolution_clock::now();
    double build_table_time_in_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    size_t memory_cost_in_bytes = 0;
    memory_cost_in_bytes = BuildTable::computeMemoryCostInBytes(query_graph, candidates_count, edge_matrix);

    start = std::chrono::high_resolution_clock::now();

    ui *matching_order = NULL;
    ui *pivots = NULL;
    ui **weight_array = NULL;

    size_t order_num = 0;
    sscanf(input_order_num.c_str(), "%zu", &order_num);
    size_t output_limit = 0;
    size_t embedding_count = 0;
    if (input_max_embedding_num == "MAX")
    {
        output_limit = std::numeric_limits<size_t>::max();
    }
    else
    {
        sscanf(input_max_embedding_num.c_str(), "%zu", &output_limit);
    }
    if (input_mode == "Test")
    {
        std::string mo = "";
        mo = "";
        // RL matching order
        ui *RL_matching_order = new ui[2 * query_graph->getVerticesCount()];
        std::stringstream ss(input_order_type);
        std::string item;
        std::vector<ui> tokens;
        while (std::getline(ss, item, '-'))
        {
            tokens.push_back(std::stoul(item));
        }
        for (uint idx = 0; idx < 2 * query_graph->getVerticesCount(); idx++)
        {
            RL_matching_order[idx] = (ui)tokens[idx];
        }
        ui *RL_pivots = new ui[2 * query_graph->getVerticesCount()];
        for (uint idx = 0; idx < 2 * query_graph->getVerticesCount(); ++idx)
        {
            std::vector<bool> visited(query_graph->getVerticesCount(), false);
            bool set = false;
            for (uint idx_pre = 0; idx_pre < idx; ++idx_pre)
            {
                if (!visited[RL_matching_order[idx_pre]])
                    visited[RL_matching_order[idx_pre]] = true;
                else
                {
                    if (query_graph->checkEdgeExistence(RL_matching_order[idx_pre], RL_matching_order[idx]))
                    {
                        RL_pivots[idx] = RL_matching_order[idx_pre];
                        set = true;
                        break;
                    }
                }
            }
            if (!set)
                RL_pivots[idx] = 0;
        }
        ui *RL_add_pool_depth = new ui[query_graph->getVerticesCount()];
        for (ui idx = 0; idx < 2 * query_graph->getVerticesCount(); ++idx)
        {
            for (ui idx_pre = 0; idx_pre < idx; idx_pre++)
            {
                if (RL_matching_order[idx] == RL_matching_order[idx_pre])
                {
                    RL_matching_order[RL_matching_order[idx]] = idx_pre;
                    break;
                }
            }
        }
        output_limit = 0;
        embedding_count = 0;
        size_t call_count = 0;
        size_t time_limit = 0;
        sscanf(input_time_limit.c_str(), "%zu", &time_limit);
        uint time_limit_in_sec = 500;
        EvaluateQuery::exit_ = false;
        sscanf(input_time_limit.c_str(), "%zu", &time_limit);
        time_limit_in_sec = 500;
        EvaluateQuery::exit_ = false;
        std::future_status status;
        size_t RL_search_node = 0;
        size_t RL_intersection_time = 0;
        std::future<size_t> future = std::async(std::launch::async, [data_graph, query_graph, candidates, candidates_count,
                                                                     RL_matching_order, RL_pivots, output_limit, RL_add_pool_depth, &RL_search_node, &RL_intersection_time]()
                                                { return EvaluateQuery::QuickSIStyleForRL(data_graph, query_graph, candidates, candidates_count,
                                                                                          RL_matching_order, RL_pivots, output_limit, RL_add_pool_depth, RL_search_node, RL_intersection_time); });
        do
        {
            status = future.wait_for(std::chrono::seconds(time_limit_in_sec));
            if (status == std::future_status::deferred)
            {
                // std::cout << "Deferred\n";
                exit(-1);
            }
            else if (status == std::future_status::timeout)
            {
                EvaluateQuery::exit_ = true;
            }
        } while (status != std::future_status::ready);
        embedding_count = future.get();
        auto end = std::chrono::high_resolution_clock::now();
        double enumeration_time_in_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout << "embedding_count: " << embedding_count << " enumeration_time_in_ns: " << enumeration_time_in_ns << std::endl; 

        delete[] RL_matching_order;
        delete[] RL_pivots;
        delete[] RL_add_pool_depth;
    }
    else
    {
        std::string mo = "";
        // traditional matching order
        GenerateQueryPlan::generateRIQueryPlan(data_graph, query_graph, matching_order, pivots);
        ui *traditional_matching_order = new ui[2 * query_graph->getVerticesCount()];
        for (uint idx = 0; idx < query_graph->getVerticesCount(); idx++)
        {
            traditional_matching_order[idx * 2] = matching_order[idx];
            traditional_matching_order[idx * 2 + 1] = matching_order[idx];
            mo += std::to_string(matching_order[idx]) + "-" + std::to_string(matching_order[idx]) + "-";
        }
        mo = mo.substr(0, mo.size() - 1);
        ui *traditional_pivots = new ui[2 * query_graph->getVerticesCount()];
        for (uint idx = 0; idx < 2 * query_graph->getVerticesCount(); ++idx)
        {
            std::vector<bool> visited(query_graph->getVerticesCount(), false);
            bool set = false;
            for (uint idx_pre = 0; idx_pre < idx; ++idx_pre)
            {
                if (!visited[traditional_matching_order[idx_pre]])
                    visited[traditional_matching_order[idx_pre]] = true;
                else
                {
                    if (query_graph->checkEdgeExistence(traditional_matching_order[idx_pre], traditional_matching_order[idx]))
                    {
                        traditional_pivots[idx] = traditional_matching_order[idx_pre];
                        set = true;
                        break;
                    }
                }
            }
            if (!set)
                traditional_pivots[idx] = 0;
        }
        ui *traditional_add_pool_depth = new ui[query_graph->getVerticesCount()];
        for (ui idx = 0; idx < 2 * query_graph->getVerticesCount(); ++idx)
        {
            for (ui idx_pre = 0; idx_pre < idx; idx_pre++)
            {
                if (traditional_matching_order[idx] == traditional_matching_order[idx_pre])
                {
                    traditional_add_pool_depth[traditional_matching_order[idx]] = idx_pre;
                    break;
                }
            }
        }

        size_t traditional_search_node = 0;
        size_t traditional_intersection_time = 0;

        size_t call_count = 0;
        size_t time_limit = 0;
        sscanf(input_time_limit.c_str(), "%zu", &time_limit);
        uint time_limit_in_sec = 500;
        EvaluateQuery::exit_ = false;
        size_t RL_search_node = 0;
        size_t RL_intersection_time = 0;
        std::future<size_t> future = std::async(std::launch::async, [data_graph, query_graph, candidates, candidates_count,
                                                                     traditional_matching_order, traditional_pivots, output_limit, traditional_add_pool_depth, &traditional_search_node, &traditional_intersection_time]()
                                                { return EvaluateQuery::QuickSIStyleForRL(data_graph, query_graph, candidates, candidates_count,
                                                                                          traditional_matching_order, traditional_pivots, output_limit, traditional_add_pool_depth, traditional_search_node, traditional_intersection_time); });
        std::future_status status;
        do
        {
            status = future.wait_for(std::chrono::seconds(time_limit_in_sec));
            if (status == std::future_status::deferred)
            {
                // std::cout << "Deferred\n";
                exit(-1);
            }
            else if (status == std::future_status::timeout)
            {
                EvaluateQuery::exit_ = true;
            }
        } while (status != std::future_status::ready);
        embedding_count = future.get();

        delete[] traditional_matching_order;
        delete[] traditional_pivots;
        delete[] traditional_add_pool_depth;

        mo = "";
        // RL matching order
        ui *RL_matching_order = new ui[2 * query_graph->getVerticesCount()];
        std::stringstream ss(input_order_type);
        std::string item;
        std::vector<ui> tokens;
        while (std::getline(ss, item, '-'))
        {
            tokens.push_back(std::stoul(item));
        }
        for (uint idx = 0; idx < 2 * query_graph->getVerticesCount(); idx++)
        {
            RL_matching_order[idx] = (ui)tokens[idx];
        }
        ui *RL_pivots = new ui[2 * query_graph->getVerticesCount()];
        for (uint idx = 0; idx < 2 * query_graph->getVerticesCount(); ++idx)
        {
            std::vector<bool> visited(query_graph->getVerticesCount(), false);
            bool set = false;
            for (uint idx_pre = 0; idx_pre < idx; ++idx_pre)
            {
                if (!visited[RL_matching_order[idx_pre]])
                    visited[RL_matching_order[idx_pre]] = true;
                else
                {
                    if (query_graph->checkEdgeExistence(RL_matching_order[idx_pre], RL_matching_order[idx]))
                    {
                        RL_pivots[idx] = RL_matching_order[idx_pre];
                        set = true;
                        break;
                    }
                }
            }
            if (!set)
                RL_pivots[idx] = 0;
        }
        ui *RL_add_pool_depth = new ui[query_graph->getVerticesCount()];
        for (ui idx = 0; idx < 2 * query_graph->getVerticesCount(); ++idx)
        {
            for (ui idx_pre = 0; idx_pre < idx; idx_pre++)
            {
                if (RL_matching_order[idx] == RL_matching_order[idx_pre])
                {
                    RL_matching_order[RL_matching_order[idx]] = idx_pre;
                    break;
                }
            }
        }
        output_limit = 0;
        embedding_count = 0;

        call_count = 0;
        time_limit = 0;
        sscanf(input_time_limit.c_str(), "%zu", &time_limit);
        time_limit_in_sec = 500;
        EvaluateQuery::exit_ = false;
        future = std::async(std::launch::async, [data_graph, query_graph, candidates, candidates_count,
                                                                     RL_matching_order, RL_pivots, output_limit, RL_add_pool_depth, &RL_search_node, &RL_intersection_time]()
                                                { return EvaluateQuery::QuickSIStyleForRL(data_graph, query_graph, candidates, candidates_count,
                                                                                          RL_matching_order, RL_pivots, output_limit, RL_add_pool_depth, RL_search_node, RL_intersection_time); });
        do
        {
            status = future.wait_for(std::chrono::seconds(time_limit_in_sec));
            if (status == std::future_status::deferred)
            {
                // std::cout << "Deferred\n";
                exit(-1);
            }
            else if (status == std::future_status::timeout)
            {
                EvaluateQuery::exit_ = true;
            }
        } while (status != std::future_status::ready);
        embedding_count = future.get();

        delete[] RL_matching_order;
        delete[] RL_pivots;
        delete[] RL_add_pool_depth;

        std::cout << traditional_search_node << " " << traditional_intersection_time << " " << RL_search_node << " " << RL_intersection_time << std::endl;
    }

    // std::cout << "--------------------------------------------------------------------" << std::endl;
    // std::cout << "Release memories..." << std::endl;
    /**
     * Release the allocated memories.
     */
    delete[] candidates_count;
    delete[] matching_order;
    delete[] pivots;
    for (ui i = 0; i < query_graph->getVerticesCount(); ++i)
    {
        delete[] candidates[i];
    }
    delete[] candidates;

    if (edge_matrix != NULL)
    {
        for (ui i = 0; i < query_graph->getVerticesCount(); ++i)
        {
            for (ui j = 0; j < query_graph->getVerticesCount(); ++j)
            {
                delete edge_matrix[i][j];
            }
            delete[] edge_matrix[i];
        }
        delete[] edge_matrix;
    }
    if (weight_array != NULL)
    {
        for (ui i = 0; i < query_graph->getVerticesCount(); ++i)
        {
            delete[] weight_array[i];
        }
        delete[] weight_array;
    }

    delete query_graph;
    delete data_graph;

    return 0;
}