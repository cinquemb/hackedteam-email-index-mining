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
#include <Eigen/SVD>
#include <Eigen/LU>
#include <Eigen/Sparse>

#include <armadillo>


/* load in isigma_ut or isigma_vt if emails trying to find documents that match query or most relevent doucument by search term*/

/* load in sparce matrix for each person and compute each message into the lower dimensional space */

/* take term and translate it into lower dimensional space using the word map and " ,"  delimiter */

/* compute the cosine distances between the terms and each message in the lower dimensional space and return map with distance */
std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

std::vector<std::string> string_split(std::string const &input) {
    std::istringstream buffer(input);
    std::vector<std::string> ret((std::istream_iterator<std::string>(buffer)), std::istream_iterator<std::string>());
    return ret;
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

std::map<std::string,int> load_word_map(std::string& word_vector_file){
	std::ifstream t(word_vector_file);
	t.seekg(0, std::ios::end);
	size_t size = t.tellg();
	std::string buffer(size, ' ');
	t.seekg(0);
	t.read(&buffer[0], size); 
	t.close();

	size_t current = 0;
	int start_index = 0;
	size_t next = std::string::npos;
	std::map<std::string,int>  words;
	do{
		next = buffer.find(" ,", current);
		words[buffer.substr(current, next - current)] = start_index;
		current = next + 2;
		++start_index;
	}while (next != std::string::npos);

	return words;
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
    std::cout << "	Dense Arma Matrix Loaded" << std::endl;
    return matrix;
}

arma::sp_fmat load_arma_sparce_matrix(std::string& data_file_name){
    std::string line;
    int line_count = 0;
    int rows = 0;
    int columns = 0;
    arma::sp_fmat matrix;
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
        		matrix.set_size(rows, columns);
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
    std::cout << "	Sparse Arma Matrix Loaded" << std::endl;
    return matrix;
}

Eigen::SparseMatrix<float> load_eigen_sparse_matrix(std::string& data_file_name){
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
    std::cout << "	Sparse Eigen Matrix Loaded" << std::endl;
    return sparseWordMatrix;
}

void search_person(std::string& person, std::string& search_query){
	std::string word_vector_file = "../word_vectors/word_vector_order_"+ person+ ".txt";
	std::string tf_doc_matrix_file = "../raw_matrices/HT_"+person+"_mail_words_matrix_raw.txt";
	std::string isigma_ut_matrix_file = "../low_dimensional_space_representation/isigma_ut/HT_"+person+"_mail_words_matrix_isigma_ut.txt";
	std::string isigma_vt_matrix_file = "../low_dimensional_space_representation/isigma_vt/HT_"+person+"_mail_words_matrix_isigma_vt.txt";

	std::map<std::string,int> words_index_map = load_word_map(word_vector_file);
	int word_vector_size = words_index_map.size();
	arma::icolvec search_query_temp_doc(word_vector_size);
	std::vector<std::string> search_words = string_split(search_query);

	Eigen::SparseMatrix<float> tf_doc_matrix = load_eigen_sparse_matrix(tf_doc_matrix_file);
	for(int i=0; i< tf_doc_matrix.outerSize();++i){
		arma::icolvec temp_doc_col_vector(word_vector_size);
		temp_doc_col_vector.zeros();
		//temp_doc_col_vector[i] = it.value();
		for(Eigen::SparseMatrix<float>::InnerIterator it(tf_doc_matrix,i); it; ++it)
			std::cout << " row:" << i << " col: " << it.col()<< " value: " << it.value()  << std::endl;
	}
	arma::fmat isigma_ut_matrix = load_dense_matrix(isigma_ut_matrix_file);
	exit(0);
}

int main(int argc, char* argv[]){
	std::string person_list_file = "../people_file_list.md";
	std::vector<std::string> person_list = load_people(person_list_file);
	std::vector<std::map<int, double> > search_result_file_indexes;
	std::string search_query = "Hi";
	for(int i=0;i<person_list.size();++i)
		search_person(person_list[i], search_query);
	//search_result_file_indexes.push_back(search_person(person_list[i], search_query));
	return 0;
}