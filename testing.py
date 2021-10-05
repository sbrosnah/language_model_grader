from numpy import median
from grade_model import grader
from my_ordered_set import ordered_set


object = grader()
human_association_path = "/Users/spencerbrosnahan/Desktop/Research/language_model_grader/SWOW-EN.complete.csv"
language_model_path = "/Users/spencerbrosnahan/Desktop/Research/codenames-warper/Game/create-small-files/small-files/revised_small_glove.6B.300d.txt"
median_rank_outfile_path = "/Users/spencerbrosnahan/Desktop/Research/language_model_grader/glove_median_ranks.txt"
h_a_test = "/Users/spencerbrosnahan/Desktop/Research/codenames-warper/Game/lm_grader/words.txt"
l_m_test = "/Users/spencerbrosnahan/Desktop/Research/codenames-warper/Game/lm_grader/lm_test.txt"
#object.load_data(human_association_path, l_m_test)

#object.get_human_associations_stats(2)
#object.output_common_words()

object.load_data(human_association_path, language_model_path, median_rank_outfile_path)
object.grade_model(5, 100)

