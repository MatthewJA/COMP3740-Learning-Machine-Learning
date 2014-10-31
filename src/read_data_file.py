import os

def read_data_file(filename):
  with open(filename) as input_file:
    experiment_type = input_file.readline().strip()
    exec(input_file.readline())
    exec(input_file.readline())
    exec(input_file.readline())
    exec(input_file.readline())
    input_file.readline()

    data = []
    for line in input_file:
      if "done" in line:
        break
      data.append(eval(line.strip()))

    return {"experiment_type": experiment_type,
            "corruption": corruption,
            "learning_rate": learning_rate,
            "hiddens": hiddens,
            "epochs": epochs,
            "data": data}

def load_all_data():
  output = []
  for datafilename in os.listdir("data"):
    if "da" in datafilename:
      try:
        data = read_data_file("data/"+datafilename)
        if data["data"]:
          output.append(data)
      except Exception:
        print "deleting data/"+datafilename
        os.remove("data/"+datafilename)

  return output

d = load_all_data()

def get_experiments(stuff):
  return [x for x in d if all(
                stuff[y] == x[y] for y in stuff)]
