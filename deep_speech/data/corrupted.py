with open("C:/Users/MaggieEzzat/Desktop/deep_speech/data/corrupted.txt") as f:
    content = f.readlines()
content = [x.strip() for x in content]
cor = []
for i in range(len(content)):
    cor.append(content[i][37:57])
print(cor)


