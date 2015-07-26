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

#include <libxml/HTMLParser.h>
#include <libxml/xpath.h>
#include <libxml/xmlreader.h>
#include <libxml/xpathInternals.h>

#include <unicode/unistr.h>
#include <unicode/ustream.h>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <Eigen/Sparse>

#include <armadillo>

std::string home_dir = std::getenv("HOME");

std::map<std::string, int> stop_words_map;
std::map<std::string, std::string> word_count_file_map;
int total_mined_emails;
long long int total_words_per_email = 0;
std::vector<std::string> files_not_mined;
std::vector<int> file_index_not_used;
std::vector<std::string> not_enough_memory_for_svd;

//experimental perameter related to how much memory svd will take and complete in a resonable time.
int system_threash_hold = 1500000;

bool try_partial_decomp = true;

template <class T1, class T2, class Pred = std::less<T2> >
struct sort_pair_second {
    bool operator()(const std::pair<T1,T2>&left, const std::pair<T1,T2>&right) {
        Pred p;
        return p(left.second, right.second);
    }
};

xmlXPathObjectPtr get_html_nodeset(xmlDocPtr doc, xmlChar *xpath){
	xmlXPathContextPtr context;
	xmlXPathObjectPtr result;

	context = xmlXPathNewContext(doc);
	result = xmlXPathEvalExpression(xpath, context);
	if(xmlXPathNodeSetIsEmpty(result->nodesetval)){
		xmlXPathFreeObject(result);
		printf("No result\n");
		return NULL;
	}
	return result;
}

struct RetrieveKey{
    template <typename T> typename T::first_type operator()(T keyValuePair) const{
    	return keyValuePair.first;
    }
};

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

void print_vector_string(std::vector<std::string>& v){
	int vector_len = v.size();
	for(int i = 0; i< vector_len-1; i++) 
		std::cout << "\"" << v[i] << "\", ";
	std::cout << "\"" << v[vector_len-1];
    std::cout << '\n';
}

void save_data(std::string file_string, std::vector<std::string> data_vector){
    int start = 0;
    std::ofstream o(file_string);
    int length_data = data_vector.size();
    for(int k = start; k < length_data; k++){
        if(k < length_data-1)
            o << data_vector[k] << " ,";
        else
            o << data_vector[k];
    }
    o.close();
}

void save_matrix(Eigen::MatrixXf& m, std::string& file_string){
	std::ofstream o(file_string);
	if(o.is_open()){
		o << m;
	}
	o.close();
}

long get_file_size(std::string filename){
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

std::string processNode(xmlTextReaderPtr reader) {
    xmlChar *name, *value;
    name = xmlTextReaderName(reader);
    if (name == NULL)
        name = xmlStrdup(BAD_CAST "--");
    value = xmlTextReaderValue(reader);

    xmlFree(name);
    if (value == NULL)
        return "";
    else {
    	icu::UnicodeString u_string((const char*)value);
    	std::ostringstream text;
    	u_string.toLower();
		text << u_string;
    	xmlFree(value);
        return text.str();
    }
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

std::vector<std::string> get_files_to_mine(std::string files_list, std::string& top_dir_path){
	std::cout << "	Loading Email Paths" << std::endl; 
    std::vector<std::string> data_file_paths;
    std::string line;
    std::ifstream in(files_list.c_str());
    if (!in.is_open()) return data_file_paths;

    while (std::getline(in,line)){
        if(line.size() > 1){
        	line.replace(line.find(top_dir_path),top_dir_path.length(),"");
        	data_file_paths.push_back(line);
        }
            
    }
    return data_file_paths;
}

std::vector<std::string> string_split(std::string const &input) {
    std::istringstream buffer(input);
    std::vector<std::string> ret((std::istream_iterator<std::string>(buffer)), std::istream_iterator<std::string>());
    return ret;
}

std::vector<std::string> extract_words_from_email(std::string &email){
	std::stringstream ss(email);
	std::string line; std::string start_end_token = "----boundary-"; std::string text_file = "Content-Type: text/plain;"; std::string html_file = "Content-Type: text/html;"; std::string message_string = ""; std::vector<std::string> temp_words;
	std::vector<std::string> words;
	int file_type = -1; //0 for text 1 for html
	int start_end_count = 0;

	while(std::getline(ss, line)){
		if(start_end_count == 1){
			if(line.find(text_file) != std::string::npos)
				file_type = 0;
			else if(line.find(html_file) != std::string::npos)
				file_type = 1;
			else{
				if(line.find(start_end_token) != std::string::npos)
					break;
				message_string += line + "\n";
			}
		}

		if(line.find(start_end_token) != std::string::npos){
			++start_end_count;
		}

		if(start_end_count == 2){
			break;
		}
	}

	if(file_type == 1){
		//std::cout << "HTML" << std::endl;
		const char *html_data = (char *)message_string.c_str();
		xmlDocPtr html_doc = htmlReadMemory(html_data, strlen(html_data), "", "utf-8", HTML_PARSE_RECOVER | HTML_PARSE_NOERROR | HTML_PARSE_NOWARNING);
		xmlTextReaderPtr reader;
		int ret;
		reader = xmlReaderWalker(html_doc);
		std::string out_text = "";
        if (reader != NULL) {
	        ret = xmlTextReaderRead(reader);
	        while (ret == 1) {
	            out_text += processNode(reader);
	            ret = xmlTextReaderRead(reader);
	        }
	        xmlFreeTextReader(reader);
	        if (ret != 0) {
	            std::cout << "	failed to parse message" << std::endl;
	        }
	    }
	    temp_words = string_split(out_text);
	    xmlFreeDoc(html_doc);	    
	}else if(file_type == 0){
		//std::cout << "TEXT" << std::endl;
		icu::UnicodeString u_string((const char*)message_string.c_str());
		std::stringstream text;
		u_string.toLower();
		text << u_string;
		while(std::getline(text, line)){
			temp_words = string_split(line);
		}
	}else{
		//std::cout << "type not detected" << std::endl;
		//std::cout << email << std::endl;
		return words;
	}

	char chars[] = "()-.!'~\"><";
	for(int i=0;i<temp_words.size();++i){
		for(int j = 0; j < strlen(chars); ++j){
			temp_words[i].erase(std::remove(temp_words[i].begin(), temp_words[i].end(), chars[j]), temp_words[i].end());
		}
		words.push_back(temp_words[i]);
	}
	return words;
}

std::vector<std::string> extract_emails_addrs_from_email(std::string &email){
	std::stringstream ss(email);
	std::string line;
	std::string stop_token = "Status: ";
	std::string email_sign = "@";
	std::string email_start_identifier = "<";
	std::string email_end_identifier = ">";

	std::vector<std::string> emails;
	std::map<std::string, int> email_map;

	while(std::getline(ss, line)){
		if(line.find(stop_token) != std::string::npos){
			break;
		}else{
			std::vector<std::string> line_vector = string_split(line);
			for(int i=0;i<line_vector.size();i++){
				std::size_t em_sign =  line_vector[i].find(email_sign);
				std::size_t em_start =  line_vector[i].find(email_start_identifier);
				std::size_t em_end =  line_vector[i].find(email_end_identifier);

				if ((em_sign != std::string::npos) && (em_start != std::string::npos) && (em_end != std::string::npos)) {
					int size_string = em_end - em_start-1;
					email_map[line_vector[i].substr(em_start+1, size_string)] = 1;
				}
			}
		}
	}
	transform(email_map.begin(), email_map.end(), back_inserter(emails), RetrieveKey());
	return emails;
}

std::string load_email(std::string &file_string){
	std::ifstream t(file_string);
	t.seekg(0, std::ios::end);
	size_t size = t.tellg();
	std::string buffer(size, ' ');
	t.seekg(0);
	t.read(&buffer[0], size); 
	return buffer;
}

void parse_mine_files(std::vector<std::string>& mail_files, std::string& top_dir_path){
	std::cout << "	Parsing Emails and Constructing Matrix" << std::endl;
	int start_vec_len = mail_files.size();

	for(int i=0; i< start_vec_len;++i){
		std::string tmp_mail_file = top_dir_path + mail_files[i];
		std::string tmp_email = load_email(tmp_mail_file);
		std::vector<std::string> emails = extract_emails_addrs_from_email(tmp_email);
		std::vector<std::string> words = extract_words_from_email(tmp_email);
		if(words.size() > 0){
			words.insert(words.end(), emails.begin(),emails.end());
			words.erase(std::remove_if(words.begin(), words.end(),[](std::string a){ return (stop_words_map.count(a) > 0);}), words.end());
			int word_list_size_for_email = words.size();
			total_words_per_email += word_list_size_for_email;
			for(int j=0; j< words.size(); ++j){
				if(word_count_file_map.count(words[j]) > 0)
					word_count_file_map[words[j]] += " " + std::to_string(i);
				else
					word_count_file_map[words[j]] = std::to_string(i);
			}
		}else{
			files_not_mined.push_back(mail_files[i]);
			file_index_not_used.push_back(i);
		}
		if( i % 10000 == 0 && i != 0)
			std::cout << "		Emails Parsed: " << i << std::endl;
	}
}

Eigen::SparseMatrix<float> construct_sparce_matrix(std::map<std::string, std::string>& word_count_file_map, int& total_mined_emails, int& avg_words_per_file, std::string& person){
	int number_of_words = word_count_file_map.size();
	int estimated_avg_words_per_file = (int)std::ceil((float)avg_words_per_file * 10);
	if(estimated_avg_words_per_file > number_of_words)
		estimated_avg_words_per_file = number_of_words;

	Eigen::SparseMatrix<float> SparceWordMatrix(number_of_words, total_mined_emails);
	
	SparceWordMatrix.reserve(Eigen::VectorXi::Constant(total_mined_emails, estimated_avg_words_per_file));
	std::vector<std::string> word_vector; 
	std::string word_vector_file = "word_vectors/word_vector_order_"+ person+ ".txt";
	transform(word_count_file_map.begin(), word_count_file_map.end(), back_inserter(word_vector), RetrieveKey());
	std::sort(word_vector.begin(), word_vector.end());
	save_data(word_vector_file, word_vector);
	std::cout << "	Sparce matrix construction" << std::endl;
	for(int i=0;i<number_of_words;++i){
		std::vector<std::string> tmp_word_counts = string_split(word_count_file_map[word_vector[i]]);
		std::map<int, int> tmp_word_counts_map;
		std::vector<int> tmp_word_counts_keys;
		for(int j = 0; j<tmp_word_counts.size();++j){
			int column;
			std::stringstream(tmp_word_counts[j]) >> column;	
			if(tmp_word_counts_map.count(column) > 0)
				tmp_word_counts_map[column] += 1;
			else
				tmp_word_counts_map[column] = 1;
		}
		transform(tmp_word_counts_map.begin(), tmp_word_counts_map.end(), back_inserter(tmp_word_counts_keys), RetrieveKey());

		int tmp_word_counts_size = tmp_word_counts_keys.size();
		if(tmp_word_counts_size > 1){
			//std::cout << "i: " << i << " tmp_word_counts_size: " << tmp_word_counts_size << std::endl;
			for(int j=0; j<tmp_word_counts_size;++j){
				SparceWordMatrix.coeffRef(i,tmp_word_counts_keys[j]) += (float)tmp_word_counts_map[tmp_word_counts_keys[j]];
			}
		}

		if( i % 2000 == 0 && i != 0)
			std::cout << "		Word rows constructed: " << i << std::endl;
		
	}
	SparceWordMatrix.makeCompressed();
	return SparceWordMatrix;
}

void row_normalize_matrix(Eigen::SparseMatrix<float>& m){
	std::cout << "	Matrix row normalization" << std::endl;
	for (int k=0; k<m.outerSize(); ++k){
		float normalize_sum_squared = 0;
		for(Eigen::SparseMatrix<float>::InnerIterator it(m,k); it; ++it)
			normalize_sum_squared += std::pow(it.value(),2);
		float normalize_denom = std::sqrt(normalize_sum_squared);
		for(Eigen::SparseMatrix<float>::InnerIterator it(m,k); it; ++it){
			float normed_value = (it.value()/normalize_denom);
			it.valueRef() = normed_value;
		}
	}
}

void construct_sparce_matrix_file_ijv(Eigen::SparseMatrix<float>& m, std::string& file_name){
	//std::cout << "Saving sparce matrix to file" << std::endl;
	FILE* s_h_w_m_f = fopen(file_name.c_str(),"w");
	fprintf(s_h_w_m_f, "%d,%d\n",m.rows(),m.cols());
	for (int k=0; k<m.outerSize(); ++k){
		for(Eigen::SparseMatrix<float>::InnerIterator it(m,k); it; ++it)
			fprintf(s_h_w_m_f, "%d,%d,%f\n",it.row(),it.col(),it.value());
	}
	fclose (s_h_w_m_f);
}

arma::sp_fmat eigen_sparce_to_sparce_matrix_armadillo(Eigen::SparseMatrix<float>& m){
	arma::sp_fmat n_m(m.rows(), m.cols());
	for (int k=0; k<m.outerSize(); ++k){
		for(Eigen::SparseMatrix<float>::InnerIterator it(m,k); it; ++it){
			n_m(it.row(),it.col()) = it.value();
		}
	}
	return n_m;
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
    Eigen::SparseMatrix<float> SparceWordMatrix(words_count, files_count);
    SparceWordMatrix.setFromTriplets(tripletList.begin(), tripletList.end());
    std::cout << "		Sparce Matrix Loaded" << std::endl;
    return SparceWordMatrix;
}

void construct_svd(Eigen::SparseMatrix<float>& lsa_matrix, std::string& out_matrix_file_u, std::string& out_matrix_file_sigma, std::string& out_matrix_file_v, std::string& person){
	try{
		//std::cout << "		before Computing svd matricies" << std::endl;
		Eigen::JacobiSVD<Eigen::MatrixXf> lsa_matrix_svd(lsa_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
		//std::cout << "		after Computing svd matricies" << std::endl;
  
		/*
			ref: ttps://en.wikipedia.org/wiki/Latent_semantic_analysis
			See how related documents j and q are in the low-dimensional space by comparing the vectors \Sigma_k \hat{\textbf{d}}_j and \Sigma_k \hat{\textbf{d}}_q (typically by cosine similarity).

			Given a query, view this as a mini document, and compare it to your documents in the low-dimensional space.

			To do the latter, you must first translate your query into the low-dimensional space. It is then intuitive that you must use the same transformation that you use on your documents:

			d_hat = inverse(sigma) * transpose(U) *  d (column vector with words in order of how they appear in original matrix)

			then find then transpose d_hat and find the eucldeian distance between other documents in lower dimensional space
		*/

		//std::cout << "		before storing U matrix" << std::endl;
		Eigen::SparseMatrix<float> m_s_u_sparce;
		{
			Eigen::MatrixXf m_s_u = lsa_matrix_svd.matrixU();
			m_s_u_sparce = m_s_u.sparseView();
		}
		
		//std::cout << "		before storing Sigma matrix" << std::endl;
		Eigen::VectorXf lsa_matrix_singular_values = lsa_matrix_svd.singularValues();
		int dims = lsa_matrix_singular_values.size();
		Eigen::SparseMatrix<float> m_s_s_sparce(dims,dims);
		for(int i=0; i<dims;++i)
			m_s_s_sparce.coeffRef(i,i) += lsa_matrix_singular_values[i];

				
		//std::cout << "		before V* Sigma matrix" << std::endl;
		Eigen::SparseMatrix<float> m_s_v_sparce;
		{
			Eigen::MatrixXf m_s_v = lsa_matrix_svd.matrixV();
			m_s_v_sparce = m_s_v.sparseView();
		}
		
		//serialize u/sigma/v matricies to txt file
		//std::cout << "		Saving svd matricies" << std::endl;
		construct_sparce_matrix_file_ijv(m_s_u_sparce, out_matrix_file_u);
		construct_sparce_matrix_file_ijv(m_s_s_sparce, out_matrix_file_sigma);
		construct_sparce_matrix_file_ijv(m_s_v_sparce, out_matrix_file_v);
	}catch (std::bad_alloc& ba){
		std::cout << "	Not enough memory for svd on: " << person << std::endl;
	}
}

void partial_svd(Eigen::SparseMatrix<float>& lsa_matrix){
	int ind_index = 0;
	unsigned long long* rowind = (unsigned long long*)malloc(sizeof(unsigned long long) * lsa_matrix.nonZeros());
	unsigned long long* colptr = (unsigned long long*)malloc(sizeof(unsigned long long) * lsa_matrix.outerSize()+1);
	for (int k=0; k<lsa_matrix.outerSize(); ++k){
		colptr[k] = ind_index;
		for(Eigen::SparseMatrix<float>::InnerIterator it(lsa_matrix,k); it; ++it){
			rowind[ind_index] = it.row();
			++ind_index;
		}
	}
	const arma::ucolvec arma_lsa_matrix_colptr(colptr, lsa_matrix.outerSize()+1, false, true);
	const arma::ucolvec arma_lsa_matrix_rowind(rowind, lsa_matrix.nonZeros(), false, true);
	const float* lsa_matrix_csc_values = lsa_matrix.valuePtr();
	const arma::fcolvec arma_lsa_matrix_csc_values(lsa_matrix_csc_values, lsa_matrix.nonZeros());

	arma::sp_fmat X(arma_lsa_matrix_rowind, arma_lsa_matrix_colptr, arma_lsa_matrix_csc_values, lsa_matrix.rows(), lsa_matrix.cols());
	std::cout << "		Eigen to Arma completed" << std::endl;
	arma::Mat<float> U;
	arma::Col<float> s;
	arma::Mat<float> V;
	bool svds_good = arma::svds(U, s, V, X, 500);
	if(rowind) free(rowind);
	if(colptr) free(colptr);
	if(!svds_good)
		std::cout << "		Partial decomp failed" << std::endl;
}

void start_mine_people(std::string& person){
	std::string out_matrix_file = "raw_matrices/HT_"+person+"_mail_words_matrix_raw.txt";
	std::string out_matrix_file_u = "u_matrices/HT_"+person+"_mail_words_matrix_u.txt";
	std::string out_matrix_file_sigma = "sigma_matrices/HT_"+person+"_mail_words_matrix_sigma.txt";
	std::string out_matrix_file_v = "v_matrices/HT_"+person+"_mail_words_matrix_v.txt";
	
	std::string files_not_mined_file = "files_not_mined/files_not_mined_"+person+".md";
	std::ifstream ht_file_check(out_matrix_file);
	if (ht_file_check.good()){
		//TODO parse raw matrix here and skip to svd operations
		ht_file_check.close();

		if(get_file_size(out_matrix_file) < system_threash_hold){
			std::cout << "IJV raw exists for: " << person << "\n 	Trying singular value decmposition for: " << person << std::endl;
			std::ifstream ht_file_check_sigma(out_matrix_file_sigma);
			if(ht_file_check_sigma.good()){
				ht_file_check_sigma.close();
				std::cout << "		SVD raw exists for: " << person << std::endl;
			}else{
				Eigen::SparseMatrix<float> lsa_matrix = load_sparce_matrix(out_matrix_file);
				construct_svd(lsa_matrix, out_matrix_file_u, out_matrix_file_sigma, out_matrix_file_v, person);
			}
		}else{
			std::cout << "IJV raw exists for: " << person << "\n 	To large for system to compute svd for: " << person  << std::endl;
			if(try_partial_decomp){
				std::cout << " 	Trying partial svd " << std::endl;
				Eigen::SparseMatrix<float> lsa_matrix = load_sparce_matrix(out_matrix_file);
				partial_svd(lsa_matrix);
			}
		}
		return;
	}else{
		std::cout << "Mining emails for " << person << std::endl;
		std::string mail_files_list = home_dir + "/HACKINGTEAMLEAK/HACKINGTEAM_MAIL/mining_scripts/mail_files_lists/mail_files_list_"+ person+ ".md";
		std::string top_dir_path = "/HACKINGTEAMLEAK/HACKINGTEAM_MAIL/";
		std::string top_dir_path_home = home_dir +"/HACKINGTEAMLEAK/HACKINGTEAM_MAIL/";
		std::vector<std::string> mail_files = get_files_to_mine(mail_files_list, top_dir_path);
		if(mail_files.size() == 0){
			std::cout << "No email paths in " << mail_files_list << std::endl;
			return;
		}
		total_mined_emails = mail_files.size();
		std::cout << "	Emails to mine: " << total_mined_emails << std::endl;
		std::string stop_words_file_list = home_dir + "/HACKINGTEAMLEAK/HACKINGTEAM_MAIL/mining_scripts/stop_words_file_list.txt";
		stop_words_map = load_stop_words(stop_words_file_list);
		std::cout << "	Total stop words: " << stop_words_map.size() << std::endl;
		parse_mine_files(mail_files, top_dir_path_home);
		std::cout << "	Total words: " << word_count_file_map.size() << std::endl;
		int avg_words_per_file = std::ceil(total_words_per_email/(float)total_mined_emails);
		std::cout << "	Average words per file: " << avg_words_per_file << std::endl;
		//words are rows, files are columns
		Eigen::SparseMatrix<float> lsa_matrix = construct_sparce_matrix(word_count_file_map, total_mined_emails, avg_words_per_file, person);

		//normalize the rows
		row_normalize_matrix(lsa_matrix);

		//save files not mined
		save_data(files_not_mined_file, files_not_mined);

		//clear variables to free memory/rested counts
		word_count_file_map.clear();
		total_words_per_email = 0;
		files_not_mined.clear();
		file_index_not_used.clear();

		//save ijv formate of matrix
		construct_sparce_matrix_file_ijv(lsa_matrix, out_matrix_file);	
	}
}
		

int main(int argc, char* argv[]){
	std::string person_list_file = "people_file_list.md";
	std::vector<std::string> person_list = load_people(person_list_file);
	for(int i=0;i<person_list.size();++i)
		start_mine_people(person_list[i]);
	return 0;
}