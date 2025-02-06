
x=12


def f1():
  print(x)

def f2():
  x = 7
  print(x)

def main():
  global x
  x = 7
  f1()
  f2()


if __name__=="__main__":
  main()