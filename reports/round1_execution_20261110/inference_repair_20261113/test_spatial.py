from spatial import *
points=np.zeros((468,2));points[[33,133]]=[30,30];points[[362,263]]=[70,30];points[[13,14,61,291]]=[50,65]
image=np.arange(100*100*3).reshape(100,100,3)
for k in range(4):
 p=turn_points(points,k,100,100);fix,_=orientation(p);assert fix==(-k)%4
 assert np.array_equal(np.rot90(np.rot90(image,k),fix),image)
assert np.array_equal(turn_points(points,0,100,100),points)
dark=np.full((30,30,3),[80,50,40],np.uint8);fit=color_fit(dark.reshape(-1,3));o=np.ones((30,30),np.uint8)
out,bad,fraction,valid=color_mask(dark,o,fit);assert valid and not bad.any() and np.array_equal(out,dark)
dark[12:14]=255;out,bad,fraction,valid=color_mask(dark,o,fit);assert valid and bad[12:14].all() and np.all(out[12:14]==[80,50,40])
print('spatial math: four rotations/inverses, zero rotation identity, dark skin retention and white strip replacement PASS')
