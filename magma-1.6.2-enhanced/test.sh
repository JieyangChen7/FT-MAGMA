#./testing/testing_dgetrf_gpu -N 16,16
#./testing/testing_dgetrf_gpu -N 10240,10240 


rm profile
touch profile
chmod 777 profile

nvprof --profile-from-start off -o profile ./testing/testing_dgeqrf -N 20480,20480

git add profile
git commit -m "updated profile"
git push origin master
