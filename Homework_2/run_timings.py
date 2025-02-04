import subprocess
from matplotlib import pyplot as plt


if __name__ == '__main__':
    executable = "./main"
    num_threads = [1, 2, 3, 4, 5, 6, 7, 8]
    num_bins = 10
    num_data = 10_000_000
    min_meas = 0.0
    max_meas = 5.0

    global_times = []
    tree_times = []

    for n in num_threads:
        execute_string = f"{executable} {n} {num_bins} {min_meas} {max_meas} {num_data}"
        result = subprocess.check_output(execute_string, shell=True)
        result = result.decode("utf-8")

        times = []
        for line in result.splitlines():
            if line.startswith("Elapsed"):
                split_line = line.split(" ")
                times.append(float(split_line[-1]))
        
        global_times.append(times[0])
        tree_times.append(times[1])
    
    plt.plot(num_threads, global_times, label="Global Time")
    plt.plot(num_threads, tree_times, label="Tree Time")
    plt.xlabel("Number of Threads")
    plt.ylabel("Time (ms)")
    plt.title("Execution Time vs Number of Threads")
    plt.legend()

    plt.savefig("timings.png")

    