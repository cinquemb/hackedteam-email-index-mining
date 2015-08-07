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

std::string home_dir = std::getenv("HOME");

std::map<std::string, int> stop_words_map;

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

template <class T1, class T2, class Pred = std::greater<T2> >
struct sort_pair_second {
    bool operator()(const std::pair<T1,T2>&left, const std::pair<T1,T2>&right) {
        Pred p;
        return p(left.second, right.second);
    }
};

std::vector<std::string> string_split(std::string const &input) {
    std::istringstream buffer(input);
    std::vector<std::string> ret((std::istream_iterator<std::string>(buffer)), std::istream_iterator<std::string>());
    return ret;
}

std::vector<std::string> filter_words(std::vector<std::string>& temp_words){
	char chars[] = "()-.!'~\"><";
	std::vector<std::string> words;
	for(int i=0;i<temp_words.size();++i){
		for(int j = 0; j < strlen(chars); ++j){
			temp_words[i].erase(std::remove(temp_words[i].begin(), temp_words[i].end(), chars[j]), temp_words[i].end());
		}
		if(stop_words_map.count(temp_words[i]) == 0)
			words.push_back(temp_words[i]);
	}
	return words;
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

std::map<std::string, int> load_stop_words(std::string stop_words_file_list){
	std::cout << "Loading Stop Words" << std::endl; 
	std::vector<std::string> stop_words_file_paths;
	std::map<std::string, int> stop_words;
    std::string line;
    std::ifstream in(stop_words_file_list.c_str());
    if (!in.is_open()) return stop_words;

    while (std::getline(in,line)){
        if(line.size() > 1)
            stop_words_file_paths.push_back(line);
    }
    in.close();

    for(int i=0; i<stop_words_file_paths.size(); i++){
    	std::ifstream ts_in(stop_words_file_paths[i].c_str());
	    if (!ts_in.is_open()) continue;

	    while (std::getline(ts_in,line)){
	        if(line.size() > 1)
	            stop_words[line] = 1;
	    }
	    ts_in.close();
    }
    return stop_words;
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

arma::mat load_dense_matrix(std::string& data_file_name){
    std::string line;
    int line_count = 0;
    int rows = 0;
    int columns = 0;
    arma::mat matrix;
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
                double v_ij = std::atof(datas[2].c_str());
                matrix(i,j) = v_ij;
            }
        }
    	++line_count;
    }
    //std::cout << "	Dense Arma Matrix Loaded" << std::endl;
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
                double v_ij = std::atof(datas[2].c_str());
                matrix(i,j) = v_ij;
            }
        }
    	++line_count;
    }
    //std::cout << "	Sparse Arma Matrix Loaded" << std::endl;
    return matrix;
}

Eigen::SparseMatrix<double> load_eigen_sparse_matrix(std::string& data_file_name){
    std::string line;
    int line_count = 0;
    int files_count = 0;
    int words_count = 0;
    std::vector<Eigen::Triplet<double> > tripletList;
    std::ifstream in(data_file_name.c_str());
    if (!in.is_open()){
        Eigen::SparseMatrix<double> sparseWordMatrix;
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
                double v_ij = std::atof(datas[2].c_str());
                Eigen::Triplet<double> triplet(i,j,v_ij);
                tripletList.push_back(triplet);
            }
        }
    	++line_count;
    }
    Eigen::SparseMatrix<double> sparseWordMatrix(words_count, files_count);
    sparseWordMatrix.setFromTriplets(tripletList.begin(), tripletList.end());
    //std::cout << "	Sparse Eigen Matrix Loaded" << std::endl;
    return sparseWordMatrix;
}

double compute_cosine_theta_distance(arma::mat& search_query_low_dimensional_space_doc_vector, arma::mat& temp_low_dimensional_space_doc_vector){
	double sum = 0.0;
    double a = 0.0;
    double b = 0.0;
	for(int i=0;i<temp_low_dimensional_space_doc_vector.n_rows;++i){
		sum += temp_low_dimensional_space_doc_vector(i,0) * search_query_low_dimensional_space_doc_vector(i,0);
        a += std::pow(temp_low_dimensional_space_doc_vector(i,0),2);
        b += std::pow(search_query_low_dimensional_space_doc_vector(i,0),2);
	}


    a = std::sqrt(a);
    b = std::sqrt(b);
	if(sum == 0){
        return 0;
    }
    else{
		double cos_theta = sum/(a*b);
		return cos_theta;
	}
}

std::vector<std::pair<int, double> > search_person(std::string& person, std::string& search_query){
	std::string word_vector_file = "../word_vectors/word_vector_order_"+ person+ ".txt";
	std::string tf_doc_matrix_file = "../raw_matrices/HT_"+person+"_mail_words_matrix_raw.txt";
    std::string sigma_matrix_file = "../sigma_matrices/HT_"+person+"_mail_words_matrix_sigma.txt";
	std::string isigma_ut_matrix_file = "../low_dimensional_space_representation/isigma_ut/HT_"+person+"_mail_words_matrix_isigma_ut.txt";
	std::string isigma_vt_matrix_file = "../low_dimensional_space_representation/isigma_vt/HT_"+person+"_mail_words_matrix_isigma_vt.txt";

	std::map<std::string,int> words_index_map = load_word_map(word_vector_file);
	int word_vector_size = words_index_map.size();
	std::vector<std::string> search_words = string_split(search_query);
	search_words = filter_words(search_words);
	std::map<int, std::string> seach_query_word_index_map;
	std::vector<std::pair<int, double> > doc_index_distance_map_vector;
	
	
	for(int i=0; i< search_words.size(); ++i){
		if(words_index_map.count(search_words[i]) > 0)
			seach_query_word_index_map[words_index_map[search_words[i]]] = search_words[i];
	}

	if(seach_query_word_index_map.size() == 0)
		return doc_index_distance_map_vector;

    Eigen::SparseMatrix<double> tf_doc_matrix = load_eigen_sparse_matrix(tf_doc_matrix_file);
    arma::mat isigma_ut_matrix = load_dense_matrix(isigma_ut_matrix_file);
    arma::mat sigma_matrix = load_dense_matrix(sigma_matrix_file);


    arma::mat search_doc_vector(word_vector_size,1);
    search_doc_vector.zeros();
    for(std::map<int,std::string>::iterator iter = seach_query_word_index_map.begin(); iter != seach_query_word_index_map.end(); ++iter){
        std::cout << iter->first << std::endl;
        search_doc_vector(iter->first,0) = 1;
    }
    arma::mat search_low_dimensional_space_doc_vector = isigma_ut_matrix * search_doc_vector;
    arma::mat sigma_search_low_dimensional_space_doc_vector = sigma_matrix * search_low_dimensional_space_doc_vector;

	for(int i=0; i< tf_doc_matrix.outerSize();++i){
		//arma::colvec temp_doc_col_vector(word_vector_size);
		arma::mat temp_doc_col_vector(word_vector_size,1);
		temp_doc_col_vector.zeros();
		bool is_empty = true;
		for(Eigen::SparseMatrix<double>::InnerIterator it(tf_doc_matrix,i); it; ++it){
			is_empty = false;
			temp_doc_col_vector((int)it.row(),0) = it.value();
		}

		if(!is_empty){
			assert(isigma_ut_matrix.n_cols == word_vector_size);
			arma::mat temp_low_dimensional_space_doc_vector = isigma_ut_matrix * temp_doc_col_vector;
            arma::mat sigma_temp_low_dimensional_space_doc_vector= sigma_matrix * temp_low_dimensional_space_doc_vector;
			double distance = compute_cosine_theta_distance(sigma_search_low_dimensional_space_doc_vector, sigma_temp_low_dimensional_space_doc_vector);
			std::pair<int,double> tmp_pair = std::make_pair(i,distance);
			doc_index_distance_map_vector.push_back(tmp_pair);
		}
	}

	std::cout << "	person: " << person << " total docs: " << word_vector_size << std::endl;
	std::sort(doc_index_distance_map_vector.begin(), doc_index_distance_map_vector.end(), sort_pair_second<int,double>());
	return doc_index_distance_map_vector;
}

int main(int argc, char* argv[]){
	std::string search_query;
	if(argc < 2){
		std::cout << "usage: ./search_emails \"query\"" << std::endl;
		return 1;
	}else{
		search_query = argv[1];
	}
	
	std::string person_list_file = "../people_file_list.md";
	std::string stop_words_file_list = "../stop_words_file_list.txt";
	std::vector<std::string> person_list = load_people(person_list_file);
	std::vector<std::vector<std::pair<int, double> > > search_result_file_indexes;
	
	if(search_query.size() == 0)
		return 1;

	stop_words_map = load_stop_words(stop_words_file_list);
	for(int i=0;i<person_list.size();++i){
		std::vector<std::pair<int, double> > person_sorted_closest_docs = search_person(person_list[i], search_query);
		if(person_sorted_closest_docs.size() > 0)
			std::cout << "	person: " << person_list[i] << " top doc index: " << person_sorted_closest_docs[0].first << " distance: " <<person_sorted_closest_docs[0].second << " num_docs: " << person_sorted_closest_docs.size() << std::endl;
		else
			std::cout << "No docs found for " << person_list[i] << std::endl;
	}
	return 0;
}