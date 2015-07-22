#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <map>
#include <cmath>
#include <iterator>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <Eigen/Sparse>

template <class T1, class T2, class Pred = std::less<T2> >
struct sort_pair_second_less {
    bool operator()(const std::pair<T1,T2>&left, const std::pair<T1,T2>&right) {
        Pred p;
        return p(left.second, right.second);
    }
};

template <class T1, class T2, class Pred = std::greater<T2> >
struct sort_pair_second_greater {
    bool operator()(const std::pair<T1,T2>&left, const std::pair<T1,T2>&right) {
        Pred p;
        return p(left.second, right.second);
    }
};

struct RetrieveKey{
    template <typename T> typename T::first_type operator()(T keyValuePair) const{
        return keyValuePair.first;
    }
};

void save_tmp_data(std::vector<float>& data_vector){
    std::string file_string = "tmp.txt";
    int start = 0;
    std::ofstream o(file_string);
    int length_data = data_vector.size();
    for(int k = start; k < length_data; k++){
        if(k < length_data-1)
            o << k << "," << data_vector[k] << "\n";
        else
            o << data_vector[k];
    }
    o.close();
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

std::vector<std::string> load_people(std::string files_list){
    std::cout << "Loading People" << std::endl; 
    std::vector<std::string> data_file_paths;
    std::string line;
    std::ifstream in(files_list.c_str());
    if (!in.is_open()) return data_file_paths;

    while (std::getline(in,line)){
        if(line.size() > 1){
            data_file_paths.push_back(line);
        }   
    }
    return data_file_paths;
}

Eigen::SparseMatrix<float> load_sparce_matrix(std::string& data_file_name){
    std::string line;
    int line_count = 0;
    int files_count = 0;
    int words_count = 0;
    std::vector<Eigen::Triplet<float> > tripletList;
    std::ifstream in(data_file_name.c_str());
    if (!in.is_open()){
        Eigen::SparseMatrix<float> SparceWordMatrix(files_count, words_count);
        return SparceWordMatrix;
    }

    while (std::getline(in,line)){
        if(line.size() > 1){
        	if(line_count == 0){
              std::vector<std::string> datas = split(line, ',');
              files_count = std::atoi(datas[0].c_str());
              words_count = std::atoi(datas[1].c_str());
              int estimation_of_entries = files_count * (int)(words_count/100);
              tripletList.reserve(estimation_of_entries);
            }else{
                std::vector<std::string> datas = split(line, ',');
                int i = std::atoi(datas[0].c_str());
                int j = std::atoi(datas[1].c_str());
                float v_ij = std::atof(datas[2].c_str());
                Eigen::Triplet<float> triplet(i,j,v_ij);
                tripletList.push_back(triplet);
            }
        }
    	++line_count;
    }
    Eigen::SparseMatrix<float> SparceWordMatrix(files_count, words_count);
    SparceWordMatrix.setFromTriplets(tripletList.begin(), tripletList.end());
    return SparceWordMatrix;
}

std::vector<std::pair<int,float> > row_distance_map(Eigen::SparseMatrix<float>& m, std::map<std::string, float>& row_value_map, int& current_row){
    std::vector<std::pair<int,float> > row_distance_pairs;
    for (int k=0; k<m.outerSize(); ++k){
        if(current_row != k){
            float tmp_distance_sum = 0.0;
            for(Eigen::SparseMatrix<float>::InnerIterator it(m,k); it; ++it){
                std::string tmp_string = std::to_string(it.row()) + "_" + std::to_string(it.col());
                if(row_value_map.count(tmp_string) > 0)
                    tmp_distance_sum += std::pow((row_value_map[tmp_string] - it.row()), 2);
                else
                    tmp_distance_sum += std::pow(row_value_map[tmp_string], 2);
            }
            float distance = std::sqrt(tmp_distance_sum);
            std::pair<int,float> tmp_pair = std::make_pair(k,distance);
            row_distance_pairs.push_back(tmp_pair);
        }
    }
    return row_distance_pairs;
}

void plot_data(std::string& file_name, int& k, std::vector<float>& v){
    save_tmp_data(v);
    std::string command = "gnuplot -e 'set terminal png; set output \""+ file_name +"\"; set xlabel \"Points\"; set ylabel \"" + std::to_string(k) +"-distance\"; set xrange [0:" + std::to_string(v.size()-1)+"]; set yrange [0:1]; plot \"tmp.txt\" u 1:2 notitle w points";
    FILE *plot_data = popen(command.c_str(), "w");
    fclose (plot_data);
}

void generate_k_nearest_neighbors_plots(Eigen::SparseMatrix<float>& m, std::string& person){
    //plot distance vs decending sorted points defined from k
    std::cout << "Computing all k distances for " << person << " with total messages of " << m.outerSize() << std::endl; 
    int estimation_of_clusters = (int)(m.outerSize()/1000);
    std::vector<std::vector<std::pair<int,float> > > point_neighbors_distance_vectors;
    std::map<int,int> exlude_vectors;
    point_neighbors_distance_vectors.reserve(m.outerSize());
    for (int i=0; i<m.outerSize(); ++i){
        std::map<std::string, float> row_value_map;
        for(Eigen::SparseMatrix<float>::InnerIterator it(m,i); it; ++it)
            row_value_map[std::to_string(it.row()) + "_" + std::to_string(it.col())]  = it.value();
        std::vector<std::pair<int,float> > tmp_r_d_pairs = row_distance_map(m, row_value_map, i);
        std::sort(tmp_r_d_pairs.begin(), tmp_r_d_pairs.end(), sort_pair_second_less<int,float>());
        float min_dist = tmp_r_d_pairs[0].second;
        float max_dist = tmp_r_d_pairs[tmp_r_d_pairs.size()-1].second;
        for(int j=0; j< tmp_r_d_pairs.size(); ++j){
            float normed_dist = (tmp_r_d_pairs[j].second - min_dist)/(max_dist-min_dist);
            tmp_r_d_pairs[j].second = normed_dist;
        }
        std::cout << "  Message rows constructed: " << i << std::endl;
        point_neighbors_distance_vectors.push_back(tmp_r_d_pairs);
    }
    /*
    std::cout << "Plotting all k distance plots for k=1 to k=" << estimation_of_clusters << " for " << person << std::endl; 
    for(int k=1;k< estimation_of_clusters; ++k){
        std::string file_name = std::to_string(k) + "-distance_plot_"+person+".png";
        std::vector<std::pair<int,float> > tmp_k_d_pairs_list;
        std::vector<float> k_distance_vectors;
        for (int i=0; i<m.outerSize(); ++i){
            std::pair<int,float> tmp_k_d_pairs = point_neighbors_distance_vectors[i][k];
            tmp_k_d_pairs_list.push_back(tmp_k_d_pairs);
        }
        std::sort(tmp_k_d_pairs_list.begin(), tmp_k_d_pairs_list.end(), sort_pair_second_greater<int,float>());
        for(int i=0; i< tmp_k_d_pairs_list.size(); ++i)
            k_distance_vectors.push_back(tmp_k_d_pairs_list[i].second);
        plot_data(file_name, k, k_distance_vectors);
    }*/
    
}

int main(int argc, char* argv[]){
    std::string person_list_file = "../people_file_list.md";
    std::vector<std::string> person_list = load_people(person_list_file);
    std::string matrix_file_name = "../HT_"+person_list[0]+"_mail_words_sparce_matrix.txt";
    std::cout << "Loading Sparce Matrix for " << person_list[0] << std::endl; 
    Eigen::SparseMatrix<float> m = load_sparce_matrix(matrix_file_name);
    generate_k_nearest_neighbors_plots(m, person_list[0]);
    return 0;
}