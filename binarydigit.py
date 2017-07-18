import cv2
import pickle







#checking if it is working or not\
'''

itemlist=[]
testList=[]

for  i in range(11,63):
    
        for j in range(1,56):
                
                with open('binary'+str(i)+"-"+str(j)+".txt",'rb') as fp:
                          if fp is None:
                                  print("nullllllllll")
                                  
                                          
                                          
                          else:
                                item=pickle.load(fp)
                                if(j<31):
                                    itemlist.append(tuple(item))
                                    
                                else:
                                    testList.append(tuple(item))
                                   






print ('item')
print(len(itemlist[0][0]))

'''






























'''

print("a ki aukaat")


for i in range(11,63):
        a=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        if(i>36):
                t=37
        else:
                t=11
        k=i-t
        a[k]=1
        for j in range(1,56):
                b=[]
                im=cv2.imread("image"+str(i)+"-"+str(j)+".png")
                #cv2.imshow('image',im)
                gray = cv2.cvtColor( im, cv2.COLOR_BGR2GRAY )
                ret,thresh1 = cv2.threshold(gray,127,1,cv2.THRESH_BINARY_INV)
                b.append(thresh1.ravel())
                #c=thresh1.ravel()
                #print("c")
                #print (c)
                #b.append(c)
                #print ("b before ")
                #print (b)
                
                b.append(a)
                #print ("b after")
                #print (b)
                with open('binary'+str(i)+"-"+str(j)+".txt",'wb') as fp:
                        pickle.dump(b,fp)

'''


'''
itr=0
itr2=0

with open('binary11-1.txt','rb') as fp:
        item=pickle.load(fp)

print(item)
'''
'''
itemlist=[]

itr=0
for  i in range(11,63):
    
        for j in range(1,56):
                
                
                with open('binary'+str(i)+"-"+str(j)+".txt",'rb') as fp:
                          if fp is None:
                                  print("nullllllllll")
                                  
                                          
                                          
                          else:
                               item=pickle.load(fp)
                               itemlist.append(tuple(item))
                                   
       
                                       
print (itemlist[0][0])                 
print (type(itemlist))
print ("itemlist 1 value ")

'''


img=cv2.imread('a2sample.jpg')
cv2.imshow('img',img)
g = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
cv2.imshow('g',g)
im=cv2.resize(img,(28,28),cv2.INTER_CUBIC)
cv2.imshow('im',im)
gray = cv2.cvtColor( im, cv2.COLOR_BGR2GRAY )
ret,thresh1 = cv2.threshold(gray,127,1,cv2.THRESH_BINARY_INV)

cv2.imshow('thresh 1',thresh1)
for t in thresh1:
    print(t)
b= thresh1.ravel()
print('##############')
print(len(b))
print ('b')
print (b)
#import pickle

with open('binarychara2.txt', 'wb') as fp:
    pickle.dump(b, fp)

'''
print('asdfkjahsdfjkhaskldfhajksdfh')
with open ('binarychara2.txt', 'rb') as fp:
   itemlist = pickle.load(fp)       

print(len(itemlist))
print ('itemlist0')
print (itemlist)
'''

'''
binaryfile  = open('binary.txt', 'w')
for item in b:
  binaryfile.write("%s" % item)

binaryfile.close()

with open('binary.txt') as f:
      content = [x.strip('\n') for x in f.readlines()]

results = [int(i) for i in content]
print ('result')
print(results)
print ('content')
print (content)
lista=[]
print('printing content iteratively')
for a in content:
    lista.append(a)
print('printitng list asfdas')
print (lista)
print('comparing the result and b')
import collections
compare = lambda x, y: collections.Counter(x) == collections.Counter(y)

print(compare(b,results))
'''
#cv2.waitKey(0)
#cv2.destroyWindow()
