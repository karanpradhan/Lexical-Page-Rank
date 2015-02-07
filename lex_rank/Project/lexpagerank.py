#Shruthi Gorantala -  shruthig@seas.upenn.edu
#Karan Pradhan - karanpr@seas.upenn.edu
#Harshitha Yenugula - yenugula@seas.upenn.edu

import nltk
import json
import os, math, operator
from nltk import FreqDist
from nltk.corpus import PlaintextCorpusReader
from nltk import sent_tokenize, word_tokenize
import pickle
from operator import itemgetter
import numpy as np


def get_all_files(path):
	files=PlaintextCorpusReader(path,'.*')
	return files.fileids()

def load_file_sentences(filename):
	string_sentence = open(filename,'r').read()
	return [sen.lower() for sen in sent_tokenize(string_sentence)]

def load_collection_sentences(path):
	res=[]
	for i in get_all_files(path):
		res.extend(load_file_sentences(os.path.join(path,i)))
	return res

def load_file_tokens(filename):
	line = open(filename,'r').read()
	return [token.lower() for token in word_tokenize(line)]
	
def load_collection_tokens(path):
	res=[]
	for i in get_all_files(path):
		res.extend(load_file_tokens(os.path.join(path,i)))
	return res

def get_idf(dir):
	df_dict = {}
	full=list(set(load_collection_tokens(dir)))
	for vocab in full: df_dict[vocab]=0
	files = get_all_files(dir); N = len(files)
	for i in files:
		toks=list(set(load_file_tokens(os.path.join(dir,i))))
		for j in toks: df_dict[j] +=1;
	idf_dict =  dict((word,math.log(N/df_dict[word])) for word in df_dict)
	
	with open('idf_dict.pickle','wb') as handle:
		pickle.dump(idf_dict,handle)

	return

def get_tf(path):   
	freqs = FreqDist(get_dir_words(path))
	word_count = sum(freqs.values())
	df_dict = dict((x, (freqs[x]+0.0)) for x in freqs)
	return df_dict

def get_tfidf(tf_dict, idf_dict):
	for i in tf_dict.keys():
		if i not in idf_dict.keys():
			idf_dict[i]=0
	tfidf = dict((word, tf_dict[word]*idf_dict[word]) for word in tf_dict)
	return tfidf

def create_feature_space(sentences):
	tokens = [word_tokenize(s) for s in sentences]
	vocabulary = set(reduce(lambda x, y: x + y, tokens))
	return dict([(voc, i) for (i, voc) in enumerate(vocabulary)])

def vectorize_w(feature_space, vocabulary,tfidf):
	vectors = [0] * len(feature_space)	
	for word in vocabulary:
		if (feature_space.has_key(word) and tfidf.has_key(word)):
			vectors[feature_space[word]] = 1*tfidf[word]
		else :
			vectors[feature_space[word]] = 0
	return vectors

def vectorize(feature_space, sentence,tfidf):
	return vectorize_w(feature_space, list(set(word_tokenize(sentence))),tfidf)



def get_dir_words(path):
	if (os.path.isdir(path)):
		return load_collection_tokens(path);
	return load_file_tokens(path)

def list_mult(A, B):
	return map(lambda (x,y): x*y, zip(A,B))


def sum_v(A):
	A2 = [a*a for a in A]
	return sum(A2)

def cosine_similarity(A, B):
	if (sum(A)*sum(B) == 0): return 0
	return float(sum(list_mult(A,B))) / math.sqrt(float(sum_v(A)*sum_v(B)))


def get_matrix(path,threshold):
	idf_dict=pickle.load(open('idf_dict.pickle','rb'))
	tf=get_tf(path)
	tfidf=get_tfidf(tf,idf_dict)
	sents=set(load_collection_sentences(path))
	sents=list(sents)
	feature_space=create_feature_space(sents)
	sent_dict={}
	pointer={}
	for i in sents:
		pointer[i]=0

	for i in sents:
		sent_dict[i]=vectorize(feature_space,i,tfidf)
	len_sens=len(sents)
	mat=np.zeros((len_sens,len_sens))
	for i,sent1 in enumerate(sents):
		for j,sent2 in enumerate(sents):
			if sent1 != sent2:
				cos=cosine_similarity(sent_dict[sent1],sent_dict[sent2])
				if cos >= threshold:
					mat[i,j]=1
		pointer[sent1]=mat[i]
	
	return pointer,mat 


def get_adjacency_dict(path,i,threshold):
        idf_dict=pickle.load(open('idf_dict.pickle','rb'))
        tfidf=pickle.load(open('./dev_tfidf/dev_'+i+'.pickle','rb'))
        sents=list(set(load_collection_sentences(path+str(i))))
        feature_space=create_feature_space(sents)
	vector_dictionary={}
	for i  in sents:
		vector_dictionary[i]=vectorize(feature_space,i,tfidf)
        cosine_similarity_dict={}
	for i in sents:
		cosine_similarity_dict[i]=[]
	for sent1 in sents:
		for sent2 in sents:
			if sent1 != sent2:
				cos=cosine_similarity(vector_dictionary[sent1],vector_dictionary[sent2])
				
				if cos>threshold:
					cosine_similarity_dict[sent1].append(sent2)
	return cosine_similarity_dict

def get_neighbours(dict_matrix,item):
	return dict_matrix[item]



def page_rank(path,sim,damp):
	sents=list(set(load_collection_sentences(path)))
	norm=float(1.0/len(sents))
	sim_norm={}
	minval=float(1.0-damp)/len(sents)
	iter_rec={}
	for i in sents:
		sim_norm[i]=norm
	for i in sents:
		iter_rec[i]=norm
	flag = 0
	for iteration in range(75):
		if flag == 1:
			break
		for j in sents:
			neighbours=get_neighbours(sim,j)
			length=len(neighbours)
			a=0
			for k in neighbours:
				b=len(get_neighbours(sim,k))
				if b!=0:
					a=a+float(iter_rec[k]/b)
							
	     		sim_norm[j]=(1-damp)*norm + damp*a
		
		error = reduce(lambda x, y: x+y, [i**2 for i in [math.fabs(a-b) for a,b in zip(iter_rec.values(),sim_norm.values())]])
		if error < 0.0001:	
			flag = 1
		for i in sents:
			iter_rec[i]=sim_norm[i]
		

	return sim_norm



'''
def page_rank(path,threshold,damp):
	dict_pointer,mat = get_matrix(path,threshold)
	v=len(dict_pointer)
	print v
	print "asdfa"
	norm=float(1)/float(v)
	vector=norm*np.ones(v)
	first=vector
	print len(first)
	
	neigh = np.ones((v,v))
	for i in range(v):
		x=sum(mat[i,:])
		if x!=0:
			neigh[i]=mat[i]/float(x)

	for i,j in enumerate(dict_pointer):
		dict_pointer[j]=first[i]
	
	damp_vector = np.ones(v)
	damp_vector = damp_vector - damp
	damp_vector = damp_vector/v
	damp_vector=np.matrix(damp_vector)
	first=np.matrix(first)
	neigh=np.matrix(neigh)	
	for i in range(0,100):
		first=damp*first*neigh
		first = first + damp_vector
	for i,j in enumerate(dict_pointer):
		dict_pointer[j]=dict_pointer[j][0,i]
	print dict_pointer	
	return dict_pointer,first
'''


if __name__ == '__main__':
	#get_idf('/home1/c/cis530/final_project/nyt_docs')
	#cos=get_adjacency_dict('/home1/c/cis530/final_project/dev_input/dev_00',0.3)
	#print cos
	#print 'negihboursdfasd'
	#print get_neighbours(cos,'a growing body of international law has in the last 10 years made it somewhat easier to reach across borders and apprehend suspects accused of torture, genocide and other crimes against humanity.')
	

	li=['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39']
	for i in li:
		#pointer,mat=page_rank('/home1/c/cis530/final_project/dev_input/dev_'+i,0.2,0.85)
		cos=get_adjacency_dict('/home1/c/cis530/final_project/dev_input/dev_',i,0.3)
		print cos
		pr=page_rank('/home1/c/cis530/final_project/dev_input/dev_'+i,cos,0.85)
		f = open('./input_summary/summary'+i+'.txt','w')
		sorted_x = sorted(pr.iteritems(),key=operator.itemgetter(1),reverse = True)
		sorted_list= [item[0] for item in sorted_x]
		length =0
		answer=[]
		for item in list(set(sorted_list)):
			f.write(item)
			answer.append(item)
			length=length+len(word_tokenize(item))
			if length >= 120 :
				break
		f.close()	
		print 'file written'

