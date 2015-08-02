#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <map>
#include <cmath>
#include <iterator>

#include <sys/stat.h>

#include <armadillo>

std::string home_dir = std::getenv("HOME");
std::string base_path = "low_dimensional_space_representation/";

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
	std::cout << "Loading people" << std::endl; 
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

arma::fmat load_dense_matrix(std::string& data_file_name){
    std::string line;
    int line_count = 0;
    int rows = 0;
    int columns = 0;
    arma::fmat matrix;
    std::ifstream in(data_file_name.c_str());
    if (!in.is_open()){
        return matrix;
    }
    while (std::getline(in,line)){
        if(line.size() > 1){
        	if(line_count == 0){
        		std::vector<std::string> datas = split(line, ',');
        		rows = std::atoi(datas[0].c_str());
        		columns = std::atoi(datas[1].c_str());
        		matrix.zeros(rows, columns);
            }else{
                std::vector<std::string> datas = split(line, ',');
                int i = std::atoi(datas[0].c_str());
                int j = std::atoi(datas[1].c_str());
                float v_ij = std::atof(datas[2].c_str());
                matrix(i,j) = v_ij;
            }
        }
    	++line_count;
    }
    return matrix;
}


void start_right_hand_creation(std::string& person){
	std::string matrix_file_u = "../u_matrices/HT_"+person+"_mail_words_matrix_u.txt";
	std::string matrix_file_sigma = "../sigma_matrices/HT_"+person+"_mail_words_matrix_sigma.txt";
	std::string matrix_file_v = "../v_matrices/HT_"+person+"_mail_words_matrix_v.txt";

	std::string isigma_ut_matrix = "isigma_ut/HT_"+person+"_mail_words_matrix_isigma_ut.txt";
	std::string isigma_vt_matrix = "isigma_vt/HT_"+person+"_mail_words_matrix_isigma_vt.txt";


    std::ifstream isigma_ut_matrix_file_check(isigma_ut_matrix);
    if (isigma_ut_matrix_file_check.good()){
        std::cout << "  RH exists for " << person << std::endl;
    }else{
        std::cout << "Reading RH for " << person << std::endl;
        arma::fmat m_sigma_i = load_dense_matrix(matrix_file_sigma);
        for(int i=0; i< m_sigma_i.n_rows; ++i){
            float inverse_value = (1/m_sigma_i(i,i));
            m_sigma_i(i,i) = inverse_value;
        }

        std::cout << "  Loading Dense Matrix U" << std::endl;
        arma::fmat m_u_t = load_dense_matrix(matrix_file_u);
        arma::inplace_trans(m_u_t);
        std::cout << "  Loading Dense Matrix V" << std::endl;
        arma::fmat m_v_t = load_dense_matrix(matrix_file_v);
        arma::inplace_trans(m_v_t);

        arma::fmat isigma_ut = m_sigma_i * m_u_t;
        arma::fmat isigma_vt = m_sigma_i * m_v_t;

        isigma_ut.save(isigma_ut_matrix,arma::raw_ascii);
        isigma_vt.save(isigma_vt_matrix,arma::raw_ascii);
    }
}

int main(int argc, char* argv[]){
	std::string person_list_file = "../people_file_list.md";
	std::vector<std::string> person_list = load_people(person_list_file);
	for(int i=0;i<person_list.size();++i)
		start_right_hand_creation(person_list[i]);
	return 0;
}