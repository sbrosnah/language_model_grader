from grade_model import grader

class statitian:
    def __init__(self):
        print("created!")
    
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
        self.__get_human_assiciations_stats_helper(num_sds)

        words_in_common, words_not_in_common = self.__compare_wordlist_to_cues()
        print("cue words in common with cm_wordlist within ", num_sds, " standard deveations: ", words_in_common, "/", len(self.__word_list))
        print("words not in common with cm_wordlist: ", words_not_in_common, "/", len(self.__word_list))