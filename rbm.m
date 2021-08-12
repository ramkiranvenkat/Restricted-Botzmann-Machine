clc
clear all
close all
% RBM Implementation
data = load('data.file');
nv = size(data,2);
nh = nv/2;

W = 0.01*randn(nh,nv);
b = zeros(nh,1);
a = zeros(nv,1);

lr = 0.02;
k = 10;
r = 2;

for epoch = 1:10
	epoch
	for itr = 1:length(data)
		v = data(itr,:)';
		vd = v;
		samples = [];
		% gibbs sampling
		for t = 1:k+r
			p_h_given_v = sigmoid(W*v + b);
			hsample = [];
			for i=1:nh
				hsample = [hsample;(p_h_given_v(i)>rand())];
			end
			vsample = [];
			p_v_given_h = sigmoid(W'*hsample + a);
			for i=1:nv
				vsample = [vsample;(p_v_given_h(i)>rand())];
			end
			if (t > k)
				samples = [samples vsample];
			end
		end
		sample_mean_W = zeros(nh,nv);
		sample_mean_a = zeros(nv,1);
		sample_mean_b = zeros(nh,1);
		for t = 1:r
			sample_mean_W = sample_mean_W + 1/r*sigmoid(W*samples(:,t) + b)*samples(:,t)';
			sample_mean_a = sample_mean_a + 1/r*samples(:,t);
			sample_mean_b = sample_mean_b + 1/r*sigmoid(W*samples(:,t) + b);
		end

		% parameter update
		W = W + lr*((sigmoid(W*vd + b))*vd' - sample_mean_W);	
		a = a + lr*(vd - sample_mean_a);
		b = b + lr*(sigmoid(W*vd + b) - sample_mean_b);
	end
end
dp = data(10,:);
figure,imshow(reshapedata(dp))
phv = sigmoid(W*dp' + b);
hsample = [];
for i=1:nh
	hsample = [hsample;(phv(i)>rand())];
end
vsample = [];
pvh = sigmoid(W'*hsample + a);
for i=1:nv
	vsample = [vsample;(pvh(i)>rand())];
end
figure,imshow(reshapedata(vsample'))



