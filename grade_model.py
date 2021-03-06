import numpy as np 
import pandas as pd
import csv
from my_ordered_set import ordered_set
import random
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split


class grader: 
    def __init__(self):
        self.__human_associations = ""
        self.__language_model = ""
        #This is a dictionary of cue words with a list of dictionaries, each of which gives us the target word and it's occurance. These are listed in descending order
        #from moset appearances to least appearances. 
        self.__association_dict = {"cue": {"target": 0}} 
        self.__ranked_associations_dict = {"cue": ["target"]}
        #The purpose is of this is to count how many times a cue appears so that we can filter out the outliars (cues with more or less than normal occurances)
        #We do this to avoid bias in our data. 
        self.__count_dict = {"cue": 0}
        #This is a list of the cm_wordlist words found in the game codenames accompanied by it's loader function.
        self.__word_list = []
        #This is necessary in the initializer because we will be using it to grade every language model no matter what. 
        self.__load_word_list()
        #This is a list of the unbiased list of words that appear around the same number of times in the human associations dataset. 
        # It doesn't really matter that it doesn't contain all the words, because we just need enoughhuman samples to deturmine if our lamguage model
        # is human enough to use
        self.__curr_list_of_words = []
        # This is a dictionary that contains every word with it's embedding vector
        self.__lm_dict = {"word": np.empty(50)}
        self.__ordered_lm_word_dict = {}
        self.median_ranks_dict = {}
        self.median_rank_outfile = ""
    
    
    def load_data(self, human_associations, language_model, median_rank_outfile):
        self.__human_associations = human_associations
        self.__language_model = language_model
        self.median_rank_outfile = median_rank_outfile
        if human_associations != "":
            self.__load_human_associations()
        if language_model != "":
            self.__load_language_model()
    
    def __load_human_associations(self):
        in_file = open(self.__human_associations, "r")
        line = in_file.readline()
        for line in in_file:
            words_list = []
            words_list = line.split(",")
            words_list = words_list[11:15]
            for i in range(len(words_list)):
                words_list[i] = words_list[i][1:(len(words_list[i]) -1)]
            if len(words_list) > 0:       
                #self.__association_dict[words_list[0]] = words_list[1:] 
                for item in words_list[1:]:
                    if item != "No more responses":
                        self.__add_item_to_dict(words_list[0], item)
                if words_list[0] in self.__count_dict:
                    self.__count_dict[words_list[0]] += 1
                else:
                    self.__count_dict[words_list[0]] = 1
        in_file.close()
        self.__rank_targets()
    
    def __add_item_to_dict(self, cue, target):
        if cue not in self.__association_dict:
            self.__association_dict[cue] = {target: 1}
        elif cue in self.__association_dict and target not in self.__association_dict[cue]:
            self.__association_dict[cue][target] = 1
        elif cue in self.__association_dict and target in self.__association_dict[cue]:
            self.__association_dict[cue][target] += 1
        else:
            print("error")

    def __rank_targets(self):
        for cue_word in self.__association_dict.keys():
                self.__ranked_associations_dict[cue_word] = sorted(self.__association_dict[cue_word], key = self.__association_dict[cue_word].get, reverse=True)
    
    def __load_word_list(self):
        path = "/Users/spencerbrosnahan/Desktop/Research/codenames-warper/Game/players/cm_wordlist.txt"
        in_file = open(path, "r")
        for word in in_file:
            self.__word_list.append(word.strip())
        in_file.close()
    
    def get_human_associations_stats(self, num_sds):
        for key in self.__count_dict:
            self.__curr_list_of_words.append(key)

        print("Stats of entire data set:\n")   
        average, standard_deviation = self.__get_human_associations_stats_helper(num_sds)
        
        words_in_common, words_not_in_common = self.__compare_wordlist_to_cues()
        print("cue words in common with cm_wordlist within ", num_sds, " standard deveations: ", words_in_common, "/", len(self.__word_list))
        print("words not in common with cm_wordlist: ", words_not_in_common, "/", len(self.__word_list), "\n\n")

        self.__update_curr_list_of_words(average, standard_deviation, num_sds)

        print("Stats within ", num_sds, " standard deviations of the mean of the dataset:\n")
        self.__get_human_associations_stats_helper(num_sds)

        words_in_common, words_not_in_common = self.__compare_wordlist_to_cues()
        print("cue words in common with cm_wordlist within ", num_sds, " standard deveations: ", words_in_common, "/", len(self.__word_list))
        print("words not in common with cm_wordlist: ", words_not_in_common, "/", len(self.__word_list))
    
    def __get_human_associations_stats_helper(self, num_sds):
        total = 0
        average = 0
        
        total = self.__find_total_count()
        min_in_dataset, max_in_dataset = self.__find_min_and_max_occurance()
        size = self.__get_size()
        average = total / size
        sum_of_squares = self.__find_sum_of_squares(average)
        variance = self.__find_variance(sum_of_squares, size)
        standard_deviation = self.__find_standard_deviation(variance)
        
        print("average appearance: ", average)
        print("minimum appearance: ", min_in_dataset)
        print("maximum appearance: ", max_in_dataset)
        print("standard deviation appearance: ", standard_deviation)

        return (average, standard_deviation)

    def __get_size(self):
        return len(self.__curr_list_of_words)
    
    def __find_standard_deviation(self, variance):
        return variance**(.5)
    
    def __find_variance(self, sum_of_squares, size):
        return sum_of_squares / (size - 1)
        
    def __find_sum_of_squares(self, average):
        sum_of_squares = 0
        for word in self.__curr_list_of_words:
            difference = 0
            diff_squared = 0
            curr_val = self.__count_dict[word]

            difference = average - curr_val
            diff_squared = difference**2
            sum_of_squares += diff_squared
        return sum_of_squares
    
    def __find_total_count(self):
        total = 0
        for word in self.__curr_list_of_words:
            total += self.__count_dict[word]
        return total
    
    def __find_min_and_max_occurance(self):
        min = 10000
        max = 0
        for key in self.__curr_list_of_words:
            if self.__count_dict[key] > max:
                max = self.__count_dict[key]
            if self.__count_dict[key] < min:
                min = self.__count_dict[key]
        return (min, max)

    def __update_curr_list_of_words(self, mean, standard_deviation, num_sds):
        self.__curr_list_of_words = []
        num_standard_deviations = num_sds
        #We might consider removing the upper limit because it may be better to have more repetitions. On the other hand, we want to avoid bias in our results. 
        upper_limit = mean + (num_standard_deviations * standard_deviation)
        lower_limit = mean - (num_standard_deviations * standard_deviation)
        for cue in self.__count_dict:
            if self.__count_dict[cue] >= lower_limit and self.__count_dict[cue] <= upper_limit:
                self.__curr_list_of_words.append(cue)
    
    def __compare_wordlist_to_cues(self):
        words_in_common = 0
        words_not_in_common = 0
        curr_set_of_words = set()

        for word in self.__curr_list_of_words:
            curr_set_of_words.add(word)

        for word in self.__word_list:
            if word in curr_set_of_words:
                words_in_common += 1
            else:
                words_not_in_common += 1

        return (words_in_common, words_not_in_common)

    def __load_language_model(self):
        in_file = open(self.__language_model, "r")
        for line in in_file:
            num = 0
            array_values = np.empty(50)
            line_values = []
            line_values = line.split(" ")
            word = line_values[0]
            array_values = np.array(line_values[1:])
            array_values = array_values.astype(np.float32)
            self.__lm_dict[word] = array_values
        

    def output_common_words(self):
        out_file = open("lm_test.txt", "w")
        for i in range(100):
            currWord = self.__curr_list_of_words[i]
            if currWord in self.__lm_dict:
                currWordVector = self.__lm_dict[currWord]
                output_line = currWord
                for num in currWordVector:
                    output_line += (" " + str(num))
                out_file.write(output_line + '\n')
        out_file.close()

    def find_kNN(self, target, k):
        set_size = 1000
        knn_set = ordered_set(1000)
        target_vec = self.__lm_dict[target]
        for word in self.__lm_dict:
            if word != target:
                curr_vec = self.__lm_dict[word]
                cosine_similarity = self.__find_cos_sim(target_vec, curr_vec)
                knn_set.push([word, cosine_similarity])
        return_list = []
        for i in range(set_size):
            return_list.append(knn_set.set[i][0])
        self.__ordered_lm_word_dict[target] = return_list
        return return_list[:k]
        

    def __find_cos_sim(self, vec1, vec2):
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def find_median_ranks(self, k, sample_size, o_f):
        out_file = open(o_f, 'w')
        for word in self.__curr_list_of_words[:sample_size]:
            if word in self.__lm_dict:
                knn = self.find_kNN(word, k)
                list_of_targets = self.__ranked_associations_dict[word][:k]
                ranks_of_association_targets_in_lm = []
                for target in list_of_targets:
                    rank = self.find_rank_of_target(target, word)
                    ranks_of_association_targets_in_lm.append(rank)
                ranks_of_association_targets_in_lm = sorted(ranks_of_association_targets_in_lm)
                ranks_of_association_targets_in_lm = np.array(ranks_of_association_targets_in_lm)
                median_rank = np.median(ranks_of_association_targets_in_lm)
                self.median_ranks_dict[word] = median_rank
                string_to_write = str(word) + " " + str(median_rank) + "\n"
                out_file.write(string_to_write)
                print(word, " ", median_rank)
        score = "score: "
        score += (self.score_model() + "\n")
        out_file.write(score)
        out_file.close()
        



    def find_rank_of_target(self, target, cue):
        rank = 0
        for word in self.__ordered_lm_word_dict[cue]:
            if word != target:
                rank += 1
            elif word == target:
                break
        return rank

    
    def grade_model(self, k, sample_size):
        self.get_human_associations_stats(2)
        random.shuffle(self.__curr_list_of_words)
        self.find_median_ranks(k, sample_size, self.median_rank_outfile)
    
    def score_model(self):
        sum = 0
        max_possible = len(self.median_ranks_dict) * 1000
        for word in self.median_ranks_dict:
            median = self.median_ranks_dict[word]
            sum += median

        score = sum / max_possible
        return score



        




    

        
    
    


