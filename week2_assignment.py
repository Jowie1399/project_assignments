my_list = []
my_list.append(10)
my_list.append(20)
my_list.append(30)
my_list.append(40)
my_list.insert(1,15)

list_2=[50,60,70]
list_3=my_list + list_2
list_3.remove(70)
list_3.sort()
print(list_3)
print(list_3.index(30))

