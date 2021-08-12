function y = reshapedata(x)
y = []
for i=1:8
	y = [y;x((i-1)*8+1:i*8)];
end
