import os
def get_immediate_subdirectories(dir):
    return [name for name in os.listdir(dir)
            if os.path.isdir(os.path.join(dir, name))]

def get_topic_thing(directory_input):
	i=39
	
	while i>21 :
		dir_name='test_'+str(i)
		s='==== Do not change these values ====\nstopFilePath = stoplist-smart-sys.txt\nperformStemming = N\nbackgroundCorpusFreqCounts = bgCounts-Giga.txt\ntopicWordCutoff = 0.1\n==== Directory to compute topic words on ====\ninputDir = /home1/c/cis530/final_project/test_input/'+dir_name+'\n==== Output File ====\noutputFile = /home1/k/karanpr/Desktop/Project/topic_words/output_'+dir_name+'\n'
		f=open('/home1/k/karanpr/Desktop/Project/topic_words/config.example','w')
		f.write(s)
		f.close()
		print('*********************'+dir_name)
		print(open('/home1/k/karanpr/Desktop/Project/topic_words/config.example').read())
		os.system('cd /home1/c/cis530/hw4/TopicWords-v2/; java -Xmx1000m TopicSignatures ~/Desktop/Project/topic_words/config.example')
		i=i-1
		
if __name__=='__main__':
	get_topic_thing('/home1/c/cis530/final_project/test_input')

