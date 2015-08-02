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

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <armadillo>

std::string home_dir = std::getenv("HOME");
std::string base_path = "low_dimensional_space_representation/";

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

void construct_sparse_matrix_file_ijv(Eigen::SparseMatrix<float>& m, std::string& file_name){
	//std::cout << "Saving sparse matrix to file" << std::endl;
	FILE* s_h_w_m_f = fopen(file_name.c_str(),"w");
	fprintf(s_h_w_m_f, "%d,%d\n",m.rows(),m.cols());
	for (int k=0; k<m.outerSize(); ++k){
		for(Eigen::SparseMatrix<float>::InnerIterator it(m,k); it; ++it)
			fprintf(s_h_w_m_f, "%d,%d,%E\n",k,it.col(),it.value());
	}
	fclose (s_h_w_m_f);
}

Eigen::MatrixXd<float> load_dense_matrix(std::string& data_file_name){
    std::string line;
    int line_count = 0;
    int rows = 0;
    int columns = 0;
    Eigen::MatrixXd matrix;
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
        		matrix.resize(rows, columns);
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
    std::cout << "		Matrix Loaded" << std::endl;
    return matrix;
}

Eigen::SparseMatrix<float> load_sparse_matrix(std::string& data_file_name){
    std::string line;
    int line_count = 0;
    int files_count = 0;
    int words_count = 0;
    std::vector<Eigen::Triplet<float> > tripletList;
    std::ifstream in(data_file_name.c_str());
    if (!in.is_open()){
        Eigen::SparseMatrix<float> sparseWordMatrix;
        return sparseWordMatrix;
    }

    while (std::getline(in,line)){
        if(line.size() > 1){
        	if(line_count == 0){
        		std::vector<std::string> datas = split(line, ',');
        		words_count = std::atoi(datas[0].c_str());
        		files_count = std::atoi(datas[1].c_str());
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
    Eigen::SparseMatrix<float> sparseWordMatrix(words_count, files_count);
    sparseWordMatrix.setFromTriplets(tripletList.begin(), tripletList.end());
    std::cout << "		Sparse Matrix Loaded" << std::endl;
    return sparseWordMatrix;
}

void start_right_hand_creation(std::string& person){
	std::string matrix_file_u = "../u_matrices/HT_"+person+"_mail_words_matrix_u.txt";
	std::string matrix_file_sigma = "../sigma_matrices/HT_"+person+"_mail_words_matrix_sigma.txt";
	std::string matrix_file_v = "../v_matrices/HT_"+person+"_mail_words_matrix_v.txt";

	std::string isigma_ut_matrix = "isigma_ut/HT_"+person+"_mail_words_matrix_isigma_ut.txt";
	std::string isigma_vt_matrix = "isigma_vt/HT_"+person+"_mail_words_matrix_isigma_vt.txt";

	Eigen::SparseMatrix<float> m_sigma_i = load_sparse_matrix(matrix_file_sigma);

	for(int i=0; i< m_sigma_i.outerSize(); ++i){
		for(Eigen::SparseMatrix<float>::InnerIterator it(m_sigma_i,i); it; ++it){
			float inverse_value = (1/it.value());
			it.valueRef() = inverse_value;
		}
	}

	Eigen::MatrixXd<float> m_u_t = load_dense_matrix(matrix_file_u);
	m_u_t.transposeInPlace();
	Eigen::MatrixXd<float> m_v_t = load_dense_matrix(matrix_file_v);
	m_v_t.transposeInPlace();

	Eigen::SparseMatrix<float> isigma_ut = m_sigma_i * m_u_t;
	Eigen::SparseMatrix<float> isigma_vt = m_sigma_i * m_v_t;

	construct_sparse_matrix_file_ijv(isigma_ut, isigma_ut_matrix);
	construct_sparse_matrix_file_ijv(isigma_vt, isigma_vt_matrix);
}

int main(int argc, char* argv[]){
	std::string person_list_file = "../people_file_list.md";
	std::vector<std::string> person_list = load_people(person_list_file);
	for(int i=0;i<person_list.size();++i)
		start_right_hand_creation(person_list[i]);
	return 0;
}