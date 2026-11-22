function result = process_round1(roiPath, outputPrefix, modelPath, isTask)
% reader_v2, precomputed ROI input. No CNN/toolbox. Real MATLAB unverified.
% modelPath is a MAT export of the supplied reader_model_dev218.json and
% exact fixed SciPy filter coefficients; no fitting or labels at inference.
assert(nargin>=4 && isscalar(isTask), 'Explicit activity flag required');
if nargin<3, modelPath=fullfile(fileparts(mfilename('fullpath')),'reader_model.mat'); end
a=load(roiPath);m=load(modelPath);r=a.roi_data;t=double(r.t(:));rgb=double(r.rgb);valid=logical(r.valid(:));
assert(size(rgb,1)==numel(t) && all(diff(t)>0));
N=300;D=sparse(1:N-2,1:N-2,1,N-2,N)+sparse(1:N-2,2:N-1,-2,N-2,N)+sparse(1:N-2,3:N,1,N-2,N);A=speye(N)+100^2*(D'*D);
pc=[];pp=[];start=0;
while start<max(0,t(end)-10+1e-8)
 if support(t,valid,start,start+10)
  q=start+(0:299)'/30;x=interp1(t,rgb,q,'linear');
  if all(isfinite(x(:))) && min(mean(x,1))>0
   p=pos_wave(x);c=chrom_wave(x,m);
   if all(isfinite(p)) && all(isfinite(c)) && std(p,1)>=1e-12 && std(c,1)>=1e-12
    pp(:,end+1)=spectrum(p,A,m);pc(:,end+1)=spectrum(c,A,m);
   end
  end
 end
 start=start+5;
end
abstain=isempty(pc);Z=zeros(numel(m.grid),1);
if ~abstain
 c=log(mean(pc,2)+1e-30);p=log(mean(pp,2)+1e-30);Z=((c-mean(c))/std(c,1)+(p-mean(p))/std(p,1))/2;
 if any(~isfinite(Z)),abstain=true;Z(:)=0;end
end
lp=log(m.prior(:)+1e-12)+double(m.beta)*Z;post=exp(lp-max(lp));post=post/sum(post);idx=min(sum(cumsum(post)<0.5)+1,numel(m.grid));
offset=m.offset_rest;if isTask,offset=m.offset_task;end
result=struct('prediction',m.grid(idx)+offset,'abstain',abstain,'valid_windows',size(pc,2),'MATLAB_verified',false);
fid=fopen([outputPrefix '.csv'],'w');fprintf(fid,'prediction,abstain,valid_windows\n%.12g,%d,%d\n',result.prediction,result.abstain,result.valid_windows);fclose(fid);
save([outputPrefix '.mat'],'result','post','Z','-v7');
end
function ok=support(t,v,s,e)
ok=numel(t)>=2 && s>=t(1)-1e-9 && e<=t(end)+1e-9;if ~ok,return;end
for i=find(~v)'
 if t(max(1,i-1))<e-1e-10 && t(min(numel(t),i+1))>s+1e-10,ok=false;return;end
end
for i=find(diff(t)>.2+1e-12)'
 if t(i)<e-1e-10 && t(i+1)>s+1e-10,ok=false;return;end
end
ok=~any(~v & t>=s-1e-10 & t<=e+1e-10);
end
function out=pos_wave(x)
out=zeros(size(x,1),1);
for e=48:size(x,1)
 ids=e-47:e;n=bsxfun(@rdivide,x(ids,:),max(mean(x(ids,:),1),1e-8));s=[0 1 -1;-2 1 1]*n';h=s(1,:)+std(s(1,:),1)/max(std(s(2,:),1),1e-8)*s(2,:);out(ids)=out(ids)+(h-mean(h))';
end
end
function out=chrom_wave(x,m)
n=size(x,1);out=zeros(n,1);starts=0:24:n-48;if starts(end)~=n-48,starts(end+1)=n-48;end
w=.5-.5*cos(2*pi*(0:47)'/47);
for s=starts
 ids=s+(1:48);v=bsxfun(@rdivide,x(ids,:),mean(x(ids,:),1));a=ff(3*v(:,1)-2*v(:,2),m.cb,m.ca,m.czi,21);b=ff(1.5*v(:,1)+v(:,2)-1.5*v(:,3),m.cb,m.ca,m.czi,21);out(ids)=out(ids)+(a-std(a,1)/std(b,1)*b).*w;
end
end
function y=ff(x,b,a,zi,pad)
ext=[2*x(1)-x(pad+1:-1:2);x;2*x(end)-x(end-1:-1:end-pad)];y=filter(b,a,ext,zi(:)*ext(1));y=flipud(y);y=filter(b,a,y,zi(:)*y(1));y=flipud(y);y=y(pad+1:end-pad);
end
function p=spectrum(x,A,m)
y=ff(x-A\x,m.rb,m.ra,m.rzi,9);w=.5-.5*cos(2*pi*(0:299)'/300);f=fft(y.*w,4096);pow=abs(f(1:2049)).^2;pow(2:end-1)=2*pow(2:end-1);p=interp1((0:2048)'*30/4096*60,pow,m.grid(:),'linear');p=p/sum(p);
end
