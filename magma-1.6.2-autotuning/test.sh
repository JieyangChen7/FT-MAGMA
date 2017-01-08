#./testing/testing_dgetrf_gpu -N 16,16 
#./testing/testing_dpotrf_gpu -N 16,16 
./testing/testing_dgeqrf_gpu -N 16,16 
#ivy K20c - 25000
#bdz K40c - 33000
#bdz C2050 - 18000
#tardis M2075 - 28000
#./testing/testing_dgetrf_gpu -N 20480,20480

#./testing/testing_dgeqrf_gpu -N 10240,10240



#./testing/testing_dpotrf_gpu -N 5120,5120
#./testing/testing_dpotrf_gpu -N 7680,7680
#./testing/testing_dpotrf_gpu -N 10240,10240
#./testing/testing_dpotrf_gpu -N 12800,12800
#./testing/testing_dpotrf_gpu -N 15360,15360
#./testing/testing_dpotrf_gpu -N 17920,17920
#./testing/testing_dpotrf_gpu -N 20480,20480
#./testing/testing_dpotrf_gpu -N 23040,23040
#./testing/testing_dpotrf_gpu -N 25600,25600
#./testing/testing_dpotrf_gpu -N 28160,28160
#./testing/testing_dpotrf_gpu -N 30720,30720
#./testing/testing_dpotrf_gpu -N 33280,33280
#./testing/testing_dpotrf_gpu -N 35840,35840
#./testing/testing_dpotrf_gpu -N 38400,38400
#./testing/testing_dpotrf_gpu -N 40960,40960


#./testing/testing_dgetrf_gpu -N 1024,1024
#./testing/testing_dgetrf_gpu -N 5120,5120
#./testing/testing_dgetrf_gpu -N 7680,7680
#./testing/testing_dgetrf_gpu -N 10240,10240
#./testing/testing_dgetrf_gpu -N 12800,12800
#./testing/testing_dgetrf_gpu -N 15360,15360
#./testing/testing_dgetrf_gpu -N 17920,17920
#./testing/testing_dgetrf_gpu -N 20480,20480
#./testing/testing_dgetrf_gpu -N 23040,23040
#./testing/testing_dgetrf_gpu -N 25600,25600
#./testing/testing_dgetrf_gpu -N 28160,28160
#./testing/testing_dgetrf_gpu -N 30720,30720


#./testing/testing_dgeqrf_gpu -N 5120,5120
#./testing/testing_dgeqrf_gpu -N 7680,7680
#./testing/testing_dgeqrf_gpu -N 10240,10240
#./testing/testing_dgeqrf_gpu -N 12800,12800
#./testing/testing_dgeqrf_gpu -N 15360,15360
#./testing/testing_dgeqrf_gpu -N 17920,17920
#./testing/testing_dgeqrf_gpu -N 20480,20480
#./testing/testing_dgeqrf_gpu -N 23040,23040



# rm profile
# touch profile
# chmod 777 profile

# nvprof --profile-from-start off -o profile ./testing/testing_dgetrf_gpu -N 20480,20480

# git add profile
# git commit -m "updated profile"
# git push origin master
