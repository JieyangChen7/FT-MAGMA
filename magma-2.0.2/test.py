#!/usr/bin/env python
import subprocess
import csv
import numpy as np
import os


def load_matrix(m, n, filename):
    #print(filename)
    f = open(filename)
    reader = csv.reader(f)
    A = []
    for row in reader:
        tmp = []
        #print(row)
        #print(n)
        for i in range(n):

            #(row[i])
            tmp.append(float(row[i]))
        A.append(tmp)
    return A

def load_time_and_energy(filename):
    f = open(filename)
    reader = csv.reader(f)
    time = 0.0
    energy = 0.0
    for row in reader:
        time = float(row[0])
        energy = float(row[1])

    return [time, energy]

def get_error_pattern(m, n, A):
    row_set = set()
    col_set = set()
    E = 1e-10
    for i in range(m):
        for j in range(n):
            if (A[i][j] > E):
                row_set.add(i)
                col_set.add(j)

    s1 = len(row_set)
    s2 = len(col_set)

    if (s1 == 0 and s2 == 0):
        return 0
    elif (s1 == 1 and s2 == 1):
        return 1
    elif (s1 == 1 or s2 == 1):
        return 2
    else:
        return 3




def analysis_dsyrk(n, k):
    before = "dsyrk-{}-{}-diff.csv".format(n,k)
    after  = "dsyrk-{}-{}-diff-after-abft.csv".format(n,k)
    time_and_energy = "dsyrk-{}-{}-time-and-energy.csv".format(n,k)

    m1 = load_matrix(n, n, before)
    m2 = load_matrix(n, n, after)

    p1 = get_error_pattern(n, n, m1)
    p2 = get_error_pattern(n, n, m2)

    [time, energy] = load_time_and_energy(time_and_energy)

    return [p1, p2, time, energy]

def analysis_dtrsm(m, n):
    before = "dtrsm-{}-{}-diff.csv".format(m,n)
    after  = "dtrsm-{}-{}-diff-after-abft.csv".format(m,n)
    time_and_energy  = "dtrsm-{}-{}-time-and-energy.csv".format(m,n)

    m1 = load_matrix(m, n, before)
    m2 = load_matrix(m, n, after)

    p1 = get_error_pattern(m, n, m1)
    p2 = get_error_pattern(m, n, m2)

    [time, energy] = load_time_and_energy(time_and_energy)

    return [p1, p2, time, energy]

def analysis_dgemm(m, n, k):
    before = "dgemm-{}-{}-{}-diff.csv".format(m,n,k)
    after  = "dgemm-{}-{}-{}-diff-after-abft.csv".format(m,n,k)
    time_and_energy  = "dgemm-{}-{}-{}-time-and-energy.csv".format(m,n,k)

    m1 = load_matrix(m, n, before)
    m2 = load_matrix(m, n, after)

    p1 = get_error_pattern(m, n, m1)
    p2 = get_error_pattern(m, n, m2)

    [time, energy] = load_time_and_energy(time_and_energy)

    return [p1, p2, time, energy]

def test_dpotrf(n, nb, repeat, v):

    cmd = ['nvidia-settings -c :0 -a \"[gpu:0]/GPUOverVoltageOffset={}\"'.format(v)]


    result_dsyrk = [0, 0, 0, 0]
    result_dtrsm = [0, 0, 0, 0]
    result_dgemm = [0, 0, 0, 0]

    result_dsyrk_after = [0, 0, 0, 0]
    result_dtrsm_after = [0, 0, 0, 0]
    result_dgemm_after = [0, 0, 0, 0]

    time_dsyrk = 0.0
    time_dtrsm = 0.0
    time_dgemm = 0.0

    energy_dsyrk = 0.0
    energy_dtrsm = 0.0
    energy_dgemm = 0.0

    for i in range(repeat):
        cmd = ['./testing/testing_dpotrf_gpu -N {}'.format(n)]
        subprocess.call(cmd, shell=True)
        for j in range(0, n-nb, nb):
            jb = min (nb, n-j)

            [p1, p2, time, energy] = analysis_dsyrk(jb, j)
            result_dsyrk[p1]       += 1
            result_dsyrk_after[p2] += 1
            time_dsyrk   += time
            energy_dsyrk += energy


            [p1, p2, time, energy] = analysis_dgemm(n-j-jb, jb, j)
            result_dgemm[p1]       += 1
            result_dgemm_after[p2] += 1
            time_dtrsm   += time
            energy_dtrsm += energy

        
            [p1, p2, time, energy] = analysis_dtrsm(n-j-jb, nb)
            result_dtrsm[p1]       += 1
            result_dtrsm_after[p2] += 1
            time_dgemm   += time
            energy_dgemm += energy

            

    for i in range(4):
        result_dsyrk[i]       /= float(repeat) * (n-nb)/nb
        result_dsyrk_after[i] /= float(repeat) * (n-nb)/nb

        result_dtrsm[i]       /= float(repeat) * (n-nb)/nb
        result_dtrsm_after[i] /= float(repeat) * (n-nb)/nb

        result_dgemm[i]       /= float(repeat) * (n-nb)/nb
        result_dgemm_after[i] /= float(repeat) * (n-nb)/nb

    print("Result:")
    print("DSYRK: {0:.2f} {1:.2f} {2:.2f} {3:.2f}".format(result_dsyrk[0], result_dsyrk[1],result_dsyrk[2],result_dsyrk[3]))
    print("DTRSM: {0:.2f} {1:.2f} {2:.2f} {3:.2f}".format(result_dtrsm[0], result_dtrsm[1],result_dtrsm[2],result_dtrsm[3]))
    print("DGEMM: {0:.2f} {1:.2f} {2:.2f} {3:.2f}".format(result_dgemm[0], result_dgemm[1],result_dgemm[2],result_dgemm[3]))

    print("DSYRK_after: {0:.2f} {1:.2f} {2:.2f} {3:.2f}".format(result_dsyrk_after[0], result_dsyrk_after[1],result_dsyrk_after[2],result_dsyrk_after[3]))
    print("DTRSM_after: {0:.2f} {1:.2f} {2:.2f} {3:.2f}".format(result_dtrsm_after[0], result_dtrsm_after[1],result_dtrsm_after[2],result_dtrsm_after[3]))
    print("DGEMM_after: {0:.2f} {1:.2f} {2:.2f} {3:.2f}".format(result_dgemm_after[0], result_dgemm_after[1],result_dgemm_after[2],result_dgemm_after[3]))

    print("DSYRK_time: {0:.2f}".format(time_dsyrk))
    print("DTRSM_time: {0:.2f}".format(time_dtrsm))
    print("DGEMM_time: {0:.2f}".format(time_dgemm))

    print("DSYRK_energy: {0:.2f}".format(energy_dsyrk))
    print("DTRSM_energy: {0:.2f}".format(energy_dtrsm))
    print("DGEMM_energy: {0:.2f}".format(energy_dgemm))


def analysis_ssyrk(n, k):
    before = "ssyrk-{}-{}-diff.csv".format(n,k)
    after  = "ssyrk-{}-{}-diff-after-abft.csv".format(n,k)
    time_and_energy  = "ssyrk-{}-{}-time-and-energy.csv".format(n,k)

    m1 = load_matrix(n, n, before)
    m2 = load_matrix(n, n, after)

    p1 = get_error_pattern(n, n, m1)
    p2 = get_error_pattern(n, n, m2)

    [time, energy] = load_time_and_energy(time_and_energy)

    return [p1, p2, time, energy]

def analysis_strsm(m, n):
    before = "strsm-{}-{}-diff.csv".format(m,n)
    after  = "strsm-{}-{}-diff-after-abft.csv".format(m,n)
    time_and_energy  = "strsm-{}-{}-time-and-energy.csv".format(m,n)

    m1 = load_matrix(m, n, before)
    m2 = load_matrix(m, n, after)

    p1 = get_error_pattern(m, n, m1)
    p2 = get_error_pattern(m, n, m2)

    [time, energy] = load_time_and_energy(time_and_energy)

    return [p1, p2, time, energy]

def analysis_sgemm(m, n, k):
    before = "sgemm-{}-{}-{}-diff.csv".format(m,n,k)
    after  = "sgemm-{}-{}-{}-diff-after-abft.csv".format(m,n,k)
    time_and_energy  = "sgemm-{}-{}-{}-time-and-energy.csv".format(m,n,k)

    m1 = load_matrix(m, n, before)
    m2 = load_matrix(m, n, after)

    p1 = get_error_pattern(m, n, m1)
    p2 = get_error_pattern(m, n, m2)

    [time, energy] = load_time_and_energy(time_and_energy)

    return [p1, p2, time, energy]

def test_spotrf(n, nb, repeat, v, mem_clock, graphics_clock, ssyrk_output_file, strsm_output_file, sgemm_output_file):

    cmd = ['nvidia-settings -c :0 -a \"[gpu:0]/GPUOverVoltageOffset={}\"'.format(v)]
    subprocess.call(cmd, shell=True)

    cmd = ['sudo nvidia-smi -pm 1 -i 0']
    subprocess.call(cmd, shell=True)

    cmd = ['sudo nvidia-smi -pl 38 -i 0']
    subprocess.call(cmd, shell=True)

    cmd = ['sudo nvidia-smi -ac {},{}'.format(mem_clock, graphics_clock)]
    subprocess.call(cmd, shell=True)

    result_ssyrk = [0, 0, 0, 0]
    result_strsm = [0, 0, 0, 0]
    result_sgemm = [0, 0, 0, 0]

    result_ssyrk_after = [0, 0, 0, 0]
    result_strsm_after = [0, 0, 0, 0]
    result_sgemm_after = [0, 0, 0, 0]

    time_ssyrk = 0.0
    time_strsm = 0.0
    time_sgemm = 0.0

    energy_ssyrk = 0.0
    energy_strsm = 0.0
    energy_sgemm = 0.0

    for i in range(repeat):
        cmd = ['./testing/testing_spotrf_gpu -N {}'.format(n)]
        subprocess.call(cmd, shell=True)
        for j in range(0, n-nb, nb):
            jb = min (nb, n-j)

            [p1, p2, time, energy] = analysis_ssyrk(jb, j)
            result_ssyrk[p1]       += 1
            result_ssyrk_after[p2] += 1
            time_ssyrk   += time
            energy_ssyrk += energy

            

            [p1, p2, time, energy] = analysis_sgemm(n-j-jb, jb, j)
            result_sgemm[p1]       += 1
            result_sgemm_after[p2] += 1
            time_sgemm   += time
            energy_sgemm += energy

        
            [p1, p2, time, energy] = analysis_strsm(n-j-jb, nb)
            result_strsm[p1]       += 1
            result_strsm_after[p2] += 1
            time_strsm   += time
            energy_strsm += energy

            

    for i in range(4):
        result_ssyrk[i]       /= float(repeat) * (n-nb)/nb
        result_ssyrk_after[i] /= float(repeat) * (n-nb)/nb

        result_strsm[i]       /= float(repeat) * (n-nb)/nb
        result_strsm_after[i] /= float(repeat) * (n-nb)/nb

        result_sgemm[i]       /= float(repeat) * (n-nb)/nb
        result_sgemm_after[i] /= float(repeat) * (n-nb)/nb


    time_ssyrk /= float(repeat)
    time_strsm /= float(repeat)
    time_sgemm /= float(repeat)

    energy_ssyrk /= float(repeat)
    energy_strsm /= float(repeat)
    energy_sgemm /= float(repeat)


    ssyrk_writer = csv.writer(ssyrk_output_file)
    strsm_writer = csv.writer(strsm_output_file)
    sgemm_writer = csv.writer(sgemm_output_file)

    ssyrk_writer.writerow([v, result_ssyrk[0], result_ssyrk[1],result_ssyrk[2],result_ssyrk[3], 
                           result_ssyrk_after[0], result_ssyrk_after[1],result_ssyrk_after[2],result_ssyrk_after[3], 
                           time_ssyrk, energy_ssyrk ])




    strsm_writer.writerow([v, result_strsm[0], result_strsm[1],result_strsm[2],result_strsm[3], 
                           result_strsm_after[0], result_strsm_after[1],result_strsm_after[2],result_strsm_after[3], 
                           time_strsm, energy_strsm ])

    sgemm_writer.writerow([v, result_sgemm[0], result_sgemm[1],result_sgemm[2],result_sgemm[3], 
                           result_sgemm_after[0], result_sgemm_after[1],result_sgemm_after[2],result_sgemm_after[3], 
                           time_sgemm, energy_sgemm ])

    ssyrk_output_file.flush()
    strsm_output_file.flush()
    sgemm_output_file.flush()


    print("Result - {}:".format(v))
    print("SSYRK: {0:.2f} {1:.2f} {2:.2f} {3:.2f}".format(result_ssyrk[0], result_ssyrk[1],result_ssyrk[2],result_ssyrk[3]))
    print("STRSM: {0:.2f} {1:.2f} {2:.2f} {3:.2f}".format(result_strsm[0], result_strsm[1],result_strsm[2],result_strsm[3]))
    print("SGEMM: {0:.2f} {1:.2f} {2:.2f} {3:.2f}".format(result_sgemm[0], result_sgemm[1],result_sgemm[2],result_sgemm[3]))

    print("SSYRK_after: {0:.2f} {1:.2f} {2:.2f} {3:.2f}".format(result_ssyrk_after[0], result_ssyrk_after[1],result_ssyrk_after[2],result_ssyrk_after[3]))
    print("STRSM_after: {0:.2f} {1:.2f} {2:.2f} {3:.2f}".format(result_strsm_after[0], result_strsm_after[1],result_strsm_after[2],result_strsm_after[3]))
    print("SGEMM_after: {0:.2f} {1:.2f} {2:.2f} {3:.2f}".format(result_sgemm_after[0], result_sgemm_after[1],result_sgemm_after[2],result_sgemm_after[3]))

    print("SSYRK_time: {0:.6f}".format(time_ssyrk))
    print("STRSM_time: {0:.6f}".format(time_strsm))
    print("SGEMM_time: {0:.6f}".format(time_sgemm))

    print("SSYRK_energy: {0:.6f}".format(energy_ssyrk))
    print("STRSM_energy: {0:.6f}".format(energy_strsm))
    print("SGEMM_energy: {0:.6f}".format(energy_sgemm))

def test_frequency(freq):
    #test_dpotrf(1024, 128, 10, 0)

    n = 4096
    ssyrk_output_filename = "ssyrk-{}-{}.csv".format(freq, n)
    strsm_output_filename = "strsm-{}-{}.csv".format(freq, n)
    sgemm_output_filename = "sgemm-{}-{}.csv".format(freq, n)

    if (os.path.exists(ssyrk_output_filename)):
        os.remove(ssyrk_output_filename)
    if (os.path.exists(strsm_output_filename)):
        os.remove(strsm_output_filename)
    if (os.path.exists(sgemm_output_filename)):
        os.remove(sgemm_output_filename)


    ssyrk_output_file = open(ssyrk_output_filename, "a")
    strsm_output_file = open(strsm_output_filename, "a")
    sgemm_output_file = open(sgemm_output_filename, "a")


    ssyrk_writer = csv.writer(ssyrk_output_file)
    strsm_writer = csv.writer(strsm_output_file)
    sgemm_writer = csv.writer(sgemm_output_file)

    header = ["voltage", "Correct", "0D", "1D", "2D", "Correct_after", "0D_after", "1D_after", "2D_after", "Time", "Energy"]
    ssyrk_writer.writerow(header)
    strsm_writer.writerow(header)
    sgemm_writer.writerow(header)


    for v in range(0, 35000, 5000):
        test_spotrf(n, 128, 10, v, 2700, freq, ssyrk_output_file, strsm_output_file, sgemm_output_file)

    ssyrk_output_file.close()
    strsm_output_file.close()
    sgemm_output_file.close()


def main():
    freq_list = [1208, 1247, 1300, 1352, 1404] #max +150
    for freq in freq_list:
        test_frequency(freq)

if __name__ == "__main__":
    main()