import tensorflow as tf
gf = tf.GraphDef()   
m_file = open('retrained_graph.pb','rb')   ##Make sure to keep the output .pb in same folder
gf.ParseFromString(m_file.read())

with open('somefile.txt', 'a') as the_file:
    for n in gf.node:
        the_file.write(n.name+'\n')

file = open('somefile.txt','r')
data = file.readlines()
print("output name = ")
print(data[len(data)-1])

print("Input name = ")
file.seek ( 0 )
print(file.readline())