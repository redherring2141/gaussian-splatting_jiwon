import sys
#f = open("./TensoRF_jiwon/measurement/lb_test10.txt", 'r')
file_name = sys.argv[1]
output_name = sys.argv[2]
f = open(file_name)

lines = f.readlines()
#print(lines)


metrics = dict()


for line in lines:
    #l = line[29:]
    l = line
    #print(line)
    #print(l)
    if "[JWLB" in l:
    #if l.find("[JW-")>=0:
        function_name = l.split(":")[0]
        value_init = l.split(":")[1]
        value = 0
        if function_name not in metrics:
            metrics[function_name]= []
        ''' 
        if "\\x" in value_init:
            value1 = value_init.split("\\x")[0]
            value2 = value_init.split("\\x")[1]
            value = value1+"."+value2
        
        if "." in value_init:
            value1 = value_init.split(".")[0]
            value2 = value_init.split(".")[1][-6:-1:1]
            value = value1+"."+value2
        '''
        if "ms" in value_init:
            value1 = value_init.split("ms")[0]
            value2 = value_init.split("ms")[1]
            value = value1
        
        #metrics[function_name].append(float(l.split(":")[1][:8]))
        '''
        print("function_name: ", function_name)
        print("value_init: ", value_init)
        print("value1: ", value1)
        print("value2: ", value2)
        print("value: ", value)
        '''
        metrics[function_name].append(float(value))

with open(output_name, "a") as summary:
    for m in metrics:
        summary.write("<{}>_{}: {}\n".format(file_name[42:], m, sum(metrics[m])))

'''
print("<Total>")
for m in metrics:
    print("{}: {}".format(m, sum(metrics[m])))

print("<Average>")
for m in metrics:
    print("{}: {}".format(m, sum(metrics[m])/len(metrics[m])))
    #print(m)
    #print(metrics[m])
    #print(sum(metrics[m])/len(metrics[m]))
    #print(len(metrics[m]))
'''
