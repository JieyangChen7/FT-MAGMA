#./testing/testing_dgetrf_gpu -N 16,16
#ivy K20c - 25000
#bdz K40c - 33000
#bdz C2050 - 18000
#tardis M2075 - 28000
#./testing/testing_dgetrf_gpu -N 10240,10240

./testing/testing_dpotrf_gpu -N 10240,10240


# rm profile
# touch profile
# chmod 777 profile

# nvprof --profile-from-start off -o profile ./testing/testing_dgetrf_gpu -N 20480,20480

# git add profile
# git commit -m "updated profile"
# git push origin master
