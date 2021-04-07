# f = open("../resources/evaluation-69.txt", "a")
# f.write("T: " + "target_str" + "\n")
# f.write("D: " + "detok_hypo_str" + "\n")
# f.close()


score_test = "BLEU = 0.08 0.2/0.1/0.1/0.0 (BP = 1.000 ratio = 6.154 hyp_len = 480 ref_len = 78"
score_test = score_test.rsplit(' = ')[1].split(" ")[0]
print("-", score_test, "-")
print(float(score_test) / 2)
