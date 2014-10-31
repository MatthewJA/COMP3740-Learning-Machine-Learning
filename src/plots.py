from read_data_file import *
import pylab

def show_experiment(rules, show_times=True, epochs=100000):
  stuff = get_experiments(rules)

  for thing in stuff:
    experimental_data = thing["data"]
    times = [x[3] for x in experimental_data][:epochs]
    accuracies = [x[2] for x in experimental_data][:epochs]
    print thing["hiddens"], thing["learning_rate"], thing["corruption"],
    print times[-1]/len(times), accuracies[-1]
    if show_times:
      pylab.plot(times, accuracies)
      pylab.xlabel("time (s)")
    else:
      pylab.plot(accuracies)
      pylab.xlabel("epoch")

    pylab.ylabel("error rate (%)")

  pylab.show()

def everything_experiment():
  show_experiment({}, True)

def show_hidden_size_experiment():
  show_experiment({"learning_rate": 0.1,
               "corruption": 0.3,
               "experiment_type":
                    "denoising autoencoder, full feedback"},
                     False,
                     20)

def show_hidden_size_experiment_2():
  show_experiment_averaging({"learning_rate": 0.1,
               "corruption": 0.3,
               "experiment_type":
                    "denoising autoencoder, full feedback"},
                    "hiddens", True, 20)


def show_learning_rate_experiment():
  show_experiment_averaging({"hiddens": 600,
               "corruption": 0.3,
               "experiment_type":
                    "denoising autoencoder, full feedback"},
                    "learning_rate", False, 20)

def show_experiment_averaging(rules, dependent, show_times = True, epochs = 10000):
  stuff = get_experiments(rules)
  x_values = sorted(list(set(test[dependent] for test in stuff)))

  for x_value in x_values:
    relevant_data = [run["data"] for run in stuff if run[dependent] == x_value]
    times = average_of_transpose([[x[3] for x in run] for run in relevant_data])[:epochs]
    print "average time: ", x_value, times[-1] / len(times)
    accuracies = average_of_transpose([[x[2] for x in run] for run in relevant_data])[:epochs]
    if show_times:
      pylab.plot(times, accuracies, label="%s = %s"%(dependent, str(x_value)))
    else:
      pylab.plot(accuracies, label="%s = %s"%(dependent, str(x_value)))

  pylab.xlabel("time (s)" if show_times else "epoch")
  pylab.ylabel("error rate (%)")
  pylab.legend(loc = "upper center")
  pylab.show()

def average_of_transpose(list_of_lists):
  return [sum(x)/len(x) for x in zip(*list_of_lists)]

show_learning_rate_experiment()