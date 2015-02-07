The project uses pickle from python to store the frequently used dictionaries for fast computation. For example idf_dict.pickle is the dictionary for the idf of the entire corpus. The folder dev_tfidf contains the tfidf dictionary for the development input. Similarly test_tfidf folder contains the tfidf dictionary for the test input. You can folder input_summary contains generated summaries for the 40 development inputs. The the folder test_summary contains generated summaries for the 40 test inputs.
The file lexpagerank.py is used for the generation of the summaries. For generating the summaries on development input we use the '/home1/c/cis530/final_project/dev_input/dev_' parameters for the get_adjacency_dictionary function, page rank functions. Similarly we use '/home1/c/cis530/final_project/test_input/dev_' as parameters while generating the summaries for the test_input folder. 
I have used the function from the homework solutions. The output of the lexpage rank will be stored in the respective test_summary or input_summary folders.
The fucntion get_adjacency_dictionary has the parameter for the threshold in the adjaceny matrix generation. It is currently set to 0.3
The function page_rank has the parameter for damping which is set to 0.85

Team :
Shruthi Gorantala shruthig@seas.upenn.edu
Karan Pradhan karanpr@seas.upenn.edu
Harshitha Yenugula yenugula@seas.upenn.edu