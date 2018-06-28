import numpy as np

#1 Read hmm
hmm_file = open('hmm.txt', 'r')

#2 Read dictionary
phone_file = open('dictionary.txt', 'r')
phone = {}

for line in phone_file :
    line = line.split()
    temp = []
    word = line[0]
    for i in range(1, len(line)) :
        temp.append(line[i])

    if not word in phone.keys() :
        phone[word] = temp
        
    else :
        phone[word] = [phone[word], temp]

# Construct word HMM
hmm = {}
mixes_key = "Mixtures"
num_mixes_key = "<NUMMIXES>"
trans_key = "trans"
num_state_key = "<NUMSTATES>"
mean_key = "<MEAN>"
variance_key = "<VARIANCE>"
gconst_key = "<GCONST>"
mix_key = "<MIXTURE>"

state = 3 
for index in range(21): 
    word_key = hmm_file.readline()[4:-2]
    hmm[word_key] = {}
   
    hmm_file.readline() #1
    
    Numstates = hmm_file.readline().split() #2
    hmm[word_key][Numstates[0]] = Numstates[1]
    
    if index==20:
        state = 1
    
    for st_index in range(state):
        s = hmm_file.readline().split()[1] #3
        hmm[word_key][s] = {}
        
        
        Num_Mixes = hmm_file.readline().split() #4
        hmm[word_key][s][Num_Mixes[0]] = Num_Mixes[1]
        hmm[word_key][s][mixes_key] = {}
        
        for i in range(10):
            
            mixture = hmm_file.readline().split() #5  #11 #17 #23 #29
      
            hmm[word_key][s][mixes_key][mixture[1]] = {}
            hmm[word_key][s][mixes_key][mixture[1]][mixture[0]] = mixture[2]
            
            #Mean
            mean_dim = hmm_file.readline().split() #6   #12 # 18 # 24 # 30
            mean_num = hmm_file.readline().split()  #7 #13 #19 # 25 # 31
            hmm[word_key][s][mixes_key][mixture[1]][mean_dim[0]] = mean_num
            
            #Variance
            variance_dim = hmm_file.readline().split() #8 #14 #20 #26 #32
            variance_num = hmm_file.readline().split()  #9 #15 #21 #27 #33
            hmm[word_key][s][mixes_key][mixture[1]][variance_dim[0]] = variance_num
            
            #GConst
            g_const = hmm_file.readline().split() #10 #16 #22 #28 #34 #40 #46 #52 #58 # 64
            hmm[word_key][s][mixes_key][mixture[1]][g_const[0]] = g_const[1]
            
    hmm_file.readline() #65
    trans_prob=[]
    
    if index != 20:
        for i in range(5):
            trans_prob.append(hmm_file.readline().split()) #66 #67
    else:
        for i in range(3):
            trans_prob.append(hmm_file.readline().split())
    
    hmm[word_key][trans_key] = trans_prob
    #ENDHMM
    hmm_file.readline()


#4 Read unigram and bigram
unigram_file = open('unigram.txt' ,'r')
unigram = {}

lines = unigram_file.read().split()
for index, line in enumerate(lines) :
    if index%2 == 0:
        unigram[line] = float(lines[index+1])



#print(unigram)

bigram_file = open('bigram.txt', 'r')
bigram = {}

for  line in bigram_file :
    line = line.split()
    bigram[line[0]] = {}

bigram_file = open('bigram.txt', 'r')
for line in bigram_file :
    line = line.split()
    bigram[line[0]][line[1]] = float(line[2])

#print(bigram)

#5 universal utterance hmm - this mean gaussian model, right?
class HMM :
    def __init__(self, hmm) :
        self.hmm = hmm
        self.phone = list(self.hmm.keys())

    def state_num(self, phone) :
        return int(self.hmm[phone][num_state_key])

    def transition_prob(self, phone) :
        return np.array(self.hmm[phone][trans_key]).astype(float)

    def states(self, phone) :
        temp = []
        
        for x in self.hmm[phone] :
            if x in [num_state_key, trans_key] :
                continue
            else :
                temp.append(int(x))

        return temp

    def n_mixes(self, phone, state) :
        return int(self.hmm[phone][str(state)][num_mixes_key])

    def gaussian(self, phone, state) :
        return self.hmm[phone][str(state)][mixes_key]

    def initial_prob(self) :
        return 1

    def observe_prob(self, x, mixtures) :
        assert len(x) == 39

        b = []
        exp_sum = 0.0
        max_log = 0.0

        for index in mixtures.keys() :
            mean = np.array(mixtures[index][mean_key]).astype(float)
            var = np.array(mixtures[index][variance_key]).astype(float)

            weight = float(mixtures[index][mix_key])
            gconst = float(mixtures[index][gconst_key])

            log = np.log(weight) - gconst/2 +np.sum((-0.5) * pow((x-mean), 2) / var)
            b.append(log)

            
        max_log = max(b)
        max_idx = np.argmax(b)

        for i in range(len(b)) :
            if i != max_idx :
                diff = b[i] - max_log
                exp_sum +=np.exp(diff)

        return max_log + np.log(1 + exp_sum)

HMM = HMM(hmm)
# mixtures = HMM.gaussian('f', 2)

# print(mixtures)

# a = 0
# print(HMM.observe_prob([a]*39, mixtures))


#6 viterbi_algorithm
def viterbi_isolated(HMM, start, data, word, phone) :
    result = {}
    feature_num = data.shape[0]
    viterbi_path = []
    phone_index_list = []
    phone_index = 0
    current_state = 1  
    
    for t in range(start, feature_num):  
        current_phone = phone[phone_index]
        
        if(t == start):
            log_emission_prob = HMM.observe_prob(data[t], HMM.gaussian(current_phone, state=current_state+1))
            viterbi_prob = log_emission_prob + np.log(HMM.initial_prob())

            viterbi_path.append(viterbi_prob)  
            phone_index_list.append(phone_index)
            continue
        

        next_possible_states = np.where(HMM.transition_prob(current_phone)[current_state] > 0)[0]
        viterbi_probs = np.zeros(shape=HMM.state_num(current_phone))

        for next_state in next_possible_states:

            over_state = (next_state + 1 == max(HMM.states(current_phone)) + 1)

            if current_phone != 'sp' and over_state:  
                if current_phone == 'sil':
                    log_emission_prob = 0
                else:
                    next_phone = phone[phone_index + 1]
                    log_emission_prob = HMM.observe_prob(data[t], HMM.gaussian(next_phone, state=2))
                
            elif current_phone == 'sp' and over_state:
                log_emission_prob = 0
            else:
                log_emission_prob = HMM.observe_prob(data[t], HMM.gaussian(current_phone, state=next_state+1))
            
            transition_prob = HMM.transition_prob(current_phone)[current_state][next_state]

            viterbi_probs[next_state] = viterbi_path[t-start-1] + log_emission_prob + np.log(transition_prob)

            if current_phone != 'sp' and current_phone !='sil' and over_state and phone[phone_index+1] == 'sp':
                viterbi_probs[next_state] += np.log(HMM.transition_prob('sp')[0][1])
        

        max_prob = max([viterbi_probs[next_state] for next_state in next_possible_states])
        current_state = np.where(viterbi_probs == max_prob)[0][0]
        viterbi_path.append(max_prob)
        phone_index_list.append(phone_index)
        
        exit_state = HMM.transition_prob(current_phone).shape[1] - 1
        if current_state == exit_state:

            phone_index += 1
            current_state = 1

            if phone_index == len(phone) or t == feature_num-1:
                result = {'t': t, 'viterbi_path': viterbi_path}
                break

        if t == feature_num -1:
            result = {'t': t, 'viterbi_path': viterbi_path}

    return result


def viterbi_continue(HMM, data, unigram, bigram, phone) :

    feature_num = data.shape[0]
    word_sequence = []
    words = sorted(list(phone.keys()))
    viterbi_sequence = []

    #print(words)

    #print('max_length',feature_num)

    t = 0
    while  True:
        #print(word_sequence)
       # print('sequnece : {}'.format(t))

        viterbi_prob = np.zeros(shape=len(words)+1)
        viterbi_t = np.zeros(shape=len(words)+1)
        if t == 0 :

            for index, word in enumerate(words) :
                if word == "zero" :
                    result = viterbi_isolated(HMM, t, data, word, phone[word][0])
                    ob_word_prob = result['viterbi_path'][-1]
                    viterbi_t[index] = result['t']
                    viterbi_prob[index] = np.log(unigram[word]) + ob_word_prob

                    result = viterbi_isolated(HMM, t, data, word, phone[word][1])
                    ob_word_prob = result['viterbi_path'][-1]
                    viterbi_t[index+1] = result['t']
                    viterbi_prob[index+1] = np.log(unigram[word]) + ob_word_prob
                else :
                    result = viterbi_isolated(HMM, t, data, word, phone[word])
                    ob_word_prob = result['viterbi_path'][-1]
                    viterbi_t[index] = result['t']
                    viterbi_prob[index] = np.log(unigram[word]) + ob_word_prob


            word_max_index = np.argmax(viterbi_prob)
            t = int(viterbi_t[word_max_index])
            #print(viterbi_prob)
            word_sequence.append(words[word_max_index if word_max_index != len(words) else word_max_index-1])
            viterbi_sequence.append(max(viterbi_prob))
            continue

        else :
            for index, word in enumerate(words) :

                pre_word = word_sequence[-1]

                if word == 'zero' :
                    result = viterbi_isolated(HMM, t, data, word, phone[word][0])
                    ob_word_prob = result['viterbi_path'][-1]
                    viterbi_t[index] = result['t']

                    if not (pre_word in bigram.keys() and word in bigram[pre_word].keys()) :
                        viterbi_prob[index] = -1e30
                    else :
                        viterbi_prob[index] = viterbi_sequence[-1] + np.log(bigram[pre_word][word])+ ob_word_prob

                    result = viterbi_isolated(HMM, t, data, word, phone[word][1])
                    ob_word_prob = result['viterbi_path'][-1]
                    viterbi_t[index+1] = result['t']

                    if not (pre_word in bigram.keys() and word in bigram[pre_word].keys()) :
                        viterbi_prob[index+1] = -1e30
                    else :
                        viterbi_prob[index+1] = viterbi_sequence[-1] + np.log(bigram[pre_word][word])+ ob_word_prob
                else :
                    result = viterbi_isolated(HMM, t, data, word, phone[word])
                    ob_word_prob = result['viterbi_path'][-1]
                    viterbi_t[index] = result['t']
                    if not (pre_word in bigram.keys() and word in bigram[pre_word].keys()) :
                        viterbi_prob[index] = -1e30
                    else :
                        viterbi_prob[index] = viterbi_sequence[-1] + np.log(bigram[pre_word][word])+ ob_word_prob

        word_max_index = np.argmax(viterbi_prob)
        t = int(viterbi_t[word_max_index])
        #viterbi_sequence.append(max(viterbi_prob))
        #print(viterbi_prob)
        
        word_sequence.append(words[word_max_index if word_max_index != len(words) else word_max_index-1])

        if t == feature_num-1 :
            break

    return word_sequence

# 7-1 read test data
import os
path = './tst'
file_names = []

for dir_path in os.listdir(path) : #f, m
    for dir_dir_path in os.listdir(path+'/'+dir_path) : #ak ~
        for test_file in os.listdir(path+'/'+dir_path+'/'+dir_dir_path) :# file_name
            file_names.append(path+'/'+dir_path+'/'+dir_dir_path +'/'+test_file)

#print(file_names)

data = {}
test_name_list = []
for test in file_names :
    #print(test)
    file = open(test)
    name = test[:-4]
    dim = [int(x) for x in file.readline().split()]

    data[name] = {'dim' : dim}
    
    features = np.ndarray((dim[0], dim[1]))

    i = -1
    while True :
        i +=1
        line = file.readline()
        if not line : break

        features[i] = np.array(line.split()).astype(float)

    data[name]['features'] = features
    test_name_list.append(name)


#7-2 ~ 4 running and output

# test_file_name = test_name_list[4]
# print(test_file_name)

# test_data = data[test_file_name]['features']

# result = viterbi_continue(HMM, test_data, unigram, bigram, phone)

#print(result)

f = open('recognized.txt','a')
f.write('#!MLF!#\n')

for test_name in test_name_list :
    f.write('"' + test_name[2:]  + '.rec' + '"\n')
    
    test_data = data[test_name]['features']

    result = viterbi_continue(HMM, test_data, unigram, bigram, phone)

    

    for temp in result :
        if temp == '<s>' :
            continue
        else :
            f.write(temp + '\n')

    f.write('.\n')
    print(test_name)
    print(result)

f.close()

#9 mean changing the language model probability? i dont know waht mean.