from pathlib import Path

root_path_resource = "/resources/swiss/"
f = open(Path(root_path_resource) / "overview/tokens_comparison.txt", "r")
diff = []

for line in f:
    diff.append(line)

diff_sorted = sorted(diff)

f = open(Path(root_path_resource) / "overview/tokens_comparison.txt", "w")
f.write("Difference: training tokens - predictions tokens")
f.write("Length: " + str(len(diff_sorted)))

for d in diff_sorted:
    f.write(d)
