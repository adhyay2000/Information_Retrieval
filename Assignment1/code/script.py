flag=True
with open('wiki_06') as f:
	with open('output.txt','w') as g:
		while True:
			c=f.read(1)
			if(not c):
				print("END OF FILE")
				break;
			elif(c=='<'):
				flag=False;
			elif(c=='>'):
				flag=True;
			elif(flag==True):
				print(c,end="");
				g.write(c);
