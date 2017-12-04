with open("balance.data") as file:
    content = file.readlines()
    f = open("balance1.data",'w')
    for i in content:
        f.write(i[::-1])