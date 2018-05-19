import numpy as np
cimport numpy as np

from cpython cimport array
import array
cimport cython
from cython.parallel import prange

import warnings

###########################
### Casting for 'long' dtypes in Windows OS

def test_long(np.ndarray[long, ndim=1] a):
	return None

def cast_np(a):
	return a.astype(long)

def cast_py(a):
	return <long> a

###########################
### Function for fitting the model

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double main_c_loop(
				double* grad_W,
				double* grad_C,
				double* grad_bu,
				double* grad_bi,
				long* shuffled_ix,
				double* Rc,
				double* Rd,
				long* ix_u,
				long* ix_i,
				long* n_rated_by_user,
				long* st_ix_user,
				double* W,
				double* C,
				double* bu,
				double* bi,
				double alpha, double lam1, double lam2, double lam3, double lam4,
				long nR, long nI,
				double step_size,
				long update_grad,
				long randomized,
				long maxthreads
			   ) nogil:
	cdef double sumR
	cdef long st_ix
	cdef double err
	cdef double rnorm
	cdef long iid_j
	cdef long uid, iid
	cdef long n_thisuser
	cdef long st_ix_imat
	cdef double b_user
	cdef double b_item
	cdef double Rj
	cdef long i, j, r

	cdef double err_tot = 0
	for r in prange(nR, schedule='static', num_threads=maxthreads):
		if randomized == 0:
			i = r
		else:
			i = shuffled_ix[r]
		uid = ix_u[i]
		iid = ix_i[i]
		st_ix_imat = nI*iid
		b_user = bu[uid]
		b_item = bi[iid]
		sumR = 0

		st_ix = st_ix_user[uid]
		n_thisuser = n_rated_by_user[uid]
		for j in range(n_thisuser):
			iid_j = st_ix_imat + ix_i[st_ix + j]
			Rj = Rd[st_ix + j]
			sumR = sumR + W[iid_j] * Rj
			sumR = sumR + C[iid_j]

		rnorm = n_thisuser**(-alpha)
		err = Rc[i] - (b_user + b_item + rnorm * sumR)
		err_tot += err*err

		if update_grad == 1:
			grad_bu[uid] += step_size * (err - lam3*b_user)
			grad_bi[iid] += step_size * (err - lam4*b_item)
			for j in range(n_thisuser):
			# for j in prange(n_thisuser, schedule='static', nogil=True):
				iid_j = st_ix_imat + ix_i[st_ix + j]
				# Diagonals should be zero
				if iid_j != iid*(nI + 1):
					Rj = Rd[st_ix + j]
					grad_W[iid_j] += step_size * (rnorm * err * Rj - lam1*W[iid_j])
					grad_C[iid_j] += step_size * (rnorm * err - lam2*C[iid_j])

	return err_tot

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double main_c_loop_single(
				double* grad_W,
				double* grad_C,
				double* grad_bu,
				double* grad_bi,
				long* shuffled_ix,
				double* Rc,
				double* Rd,
				long* ix_u,
				long* ix_i,
				long* n_rated_by_user,
				long* st_ix_user,
				double* W,
				double* C,
				double* bu,
				double* bi,
				double alpha, double lam1, double lam2, double lam3, double lam4,
				long nR, long nI,
				double step_size,
				long update_grad,
				long randomized
			   ):
	cdef double sumR
	cdef long st_ix
	cdef double err
	cdef double rnorm
	cdef long iid_j
	cdef long uid, iid
	cdef long n_thisuser
	cdef long st_ix_imat
	cdef double b_user
	cdef double b_item
	cdef double Rj
	cdef long i, j, r

	cdef double err_tot = 0
	for r in range(nR):
		if randomized == 0:
			i = r
		else:
			i = shuffled_ix[r]
		uid = ix_u[i]
		iid = ix_i[i]
		st_ix_imat = nI*iid
		b_user = bu[uid]
		b_item = bi[iid]
		sumR = 0

		st_ix = st_ix_user[uid]
		n_thisuser = n_rated_by_user[uid]
		for j in range(n_thisuser):
			iid_j = st_ix_imat + ix_i[st_ix + j]
			Rj = Rd[st_ix + j]
			sumR = sumR + W[iid_j] * Rj
			sumR = sumR + C[iid_j]

		rnorm = n_thisuser**(-alpha)
		err = Rc[i] - (b_user + b_item + rnorm * sumR)
		err_tot += err*err

		if update_grad == 1:
			grad_bu[uid] += step_size * (err - lam3*b_user)
			grad_bi[iid] += step_size * (err - lam4*b_item)
			for j in range(n_thisuser):
			# for j in prange(n_thisuser, schedule='static', nogil=True):
				iid_j = st_ix_imat + ix_i[st_ix + j]
				# Diagonals should be zero
				if iid_j != iid*(nI + 1):
					Rj = Rd[st_ix + j]
					grad_W[iid_j] += step_size * (rnorm * err * Rj - lam1*W[iid_j])
					grad_C[iid_j] += step_size * (rnorm * err - lam2*C[iid_j])

	return err_tot

##############################
### Functions for prediction time

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double predict_item_known_user(long uid, long iid,
			double* W,
			double* C,
			double* bu,
			double* bi,
			double* Rd,
			long* ix_i,
			long* st_ix_user,
			long* n_rated_by_user,
			long nI, double alpha
			):
	cdef double rhat = 0
	cdef long st_ix = st_ix_user[uid]
	cdef long j
	cdef long r

	cdef long n_thisuser = n_rated_by_user[uid]
	for r in range(n_thisuser):
		j = st_ix + r
		iid_j = nI*iid + ix_i[j]
		rhat += C[iid_j] + W[iid_j] * Rd[j]
	return rhat*(n_thisuser**(-alpha)) + bu[uid] + bi[iid]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void predict_all_known_user(
			double* W,
			double* C,
			double* bu,
			double* bi,
			double* Rd,
			long* ix_i,
			long* st_ix_user,
			long* n_rated_by_user,
			double* pred_arr,
			long nI, long uid, double alpha,
			long nthreads
			) nogil:

	cdef long st_ix = st_ix_user[uid]
	cdef long iid
	cdef long j
	cdef long i
	cdef long iid_j
	cdef long st_ix_imat

	for i in prange(nI, schedule='static', num_threads=nthreads):
		iid = ix_i[st_ix + i]
		st_ix_imat = iid*nI
		for j in range(n_rated_by_user[uid]):
			iid_j = st_ix_imat + ix_i[st_ix + j]
			pred_arr[i] += C[iid_j] + W[iid_j] * Rd[st_ix + j]
		pred_arr[i] *= n_rated_by_user[uid]**(-alpha)
		pred_arr[i] += bu[uid] + bi[iid]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void score_all_from_rating_lst(
			double* W,
			double* C,
			double* bi,
			double* Rd,
			long* ix_ratings,
			double* pred_arr,
			long nI, long n_thisuser, double alpha,
			long nthreads
			) nogil:
	
	cdef long i
	cdef long j
	cdef long iid_j

	for i in prange(nI, schedule='static', num_threads=nthreads):
		for j in range(n_thisuser):
			iid_j = i*nI + ix_ratings[j]
			pred_arr[i] += C[iid_j] + W[iid_j] * Rd[j]
		pred_arr[i] *= n_thisuser**(-alpha)
		pred_arr[i] += bi[i]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void predict_lst(
		double* W,
		double* C,
		double* bu,
		double* bi,
		double* Rd,
		long* ix_i,
		long* n_rated_by_user,
		long* st_ix_user,
		double* pred_arr,
		long* pred_uids,
		long* pred_iids,
		long nPredict,
		long nI,
		double alpha,
		long nthreads
		) nogil:
	
	cdef long iid
	cdef long uid
	cdef long n_thisuser
	cdef long i
	cdef long j
	cdef long iid_j
	cdef long st_ix

	for i in prange(nPredict, schedule='static', num_threads=nthreads):
		uid = pred_uids[i]
		iid = pred_iids[i]
		st_ix = st_ix_user[uid]
		n_thisuser = n_rated_by_user[uid]
		for j in range(n_thisuser):
			iid_j = iid*nI + ix_i[st_ix + j]
			pred_arr[i] += C[iid_j] + W[iid_j] * Rd[st_ix + j]
		pred_arr[i] *= n_thisuser**(-alpha)
		pred_arr[i] += bu[uid] + bi[iid]


########################################
#### Python functions for prediction

def recommend_from_ratings(
		np.ndarray[double, ndim=1] empty_arr,
		np.ndarray[double, ndim=1] W,
		np.ndarray[double, ndim=1] C,
		np.ndarray[double, ndim=1] bi,
		np.ndarray[double, ndim=1] Rd,
		np.ndarray[long, ndim=1] rated_this_user,
		long n_thisuser, long nI, double alpha,
		long nthreads
	):
	score_all_from_rating_lst(
			&W[0],
			&C[0],
			&bi[0],
			&Rd[0],
			&rated_this_user[0],
			&empty_arr[0],
			nI, n_thisuser, alpha,
			nthreads
			)

def recommend_all_known_user(
	np.ndarray[double, ndim=1] empty_arr,
	np.ndarray[double, ndim=1] W,
	np.ndarray[double, ndim=1] C,
	np.ndarray[double, ndim=1] bu,
	np.ndarray[double, ndim=1] bi,
	np.ndarray[double, ndim=1] Rd,
	np.ndarray[long, ndim=1] ix_i,
	np.ndarray[long, ndim=1] n_rated_by_user,
	np.ndarray[long, ndim=1] st_ix_user,
	long nI, long uid, double alpha,
	long nthreads
	):
	
	predict_all_known_user(
			&W[0],
			&C[0],
			&bu[0],
			&bi[0],
			&Rd[0],
			&ix_i[0],
			&st_ix_user[0],
			&n_rated_by_user[0],
			&empty_arr[0],
			nI, uid, alpha,
			nthreads)

def predict_single_known_user(
	np.ndarray[double, ndim=1] W,
	np.ndarray[double, ndim=1] C,
	np.ndarray[double, ndim=1] bu,
	np.ndarray[double, ndim=1] bi,
	np.ndarray[double, ndim=1] Rd,
	np.ndarray[long, ndim=1] ix_i,
	np.ndarray[long, ndim=1] st_ix_user,
	np.ndarray[long, ndim=1] n_rated_by_user,
	long nI, long uid, long iid, double alpha
	):
	return predict_item_known_user(uid, iid,
			&W[0],
			&C[0],
			&bu[0],
			&bi[0],
			&Rd[0],
			&ix_i[0],
			&st_ix_user[0],
			&n_rated_by_user[0],
			nI, alpha
			   		)

def predict_test_set(
		np.ndarray[double, ndim=1] W,
		np.ndarray[double, ndim=1] C,
		np.ndarray[double, ndim=1] bu,
		np.ndarray[double, ndim=1] bi,
		np.ndarray[double, ndim=1] Rd,
		np.ndarray[long, ndim=1] ix_i,
		np.ndarray[long, ndim=1] n_rated_by_user,
		np.ndarray[long, ndim=1] st_ix_user,
		np.ndarray[double, ndim=1] empty_arr,
		np.ndarray[long, ndim=1] pred_uids,
		np.ndarray[long, ndim=1] pred_iids,
		long nPredict, long nI, double alpha,
		long nthreads):
	predict_lst(
		&W[0],
		&C[0],
		&bu[0],
		&bi[0],
		&Rd[0],
		&ix_i[0],
		&n_rated_by_user[0],
		&st_ix_user[0],
		&empty_arr[0],
		&pred_uids[0],
		&pred_iids[0],
		nPredict,
		nI,
		alpha,
		nthreads
	)

########################################
#### Python functions for model fitting

def calc_obj(np.ndarray[double, ndim=1] xlong, *args):
	cdef np.ndarray[double, ndim=1] Rc = args[0]
	cdef np.ndarray[double, ndim=1] Rd = args[1]
	cdef np.ndarray[long, ndim=1] ix_u = args[2]
	cdef np.ndarray[long, ndim=1] ix_i = args[3]
	cdef np.ndarray[long, ndim=1] n_rated_by_user = args[4]
	cdef np.ndarray[long, ndim=1] st_ix_user = args[5]
	cdef long nR = args[6]
	cdef long nU = args[7]
	cdef long nI = args[8]
	cdef double alpha = args[9]
	cdef double lam1 = args[10]
	cdef double lam2 = args[11]
	cdef double lam3 = args[12]
	cdef double lam4 = args[13]
	cdef long nthreads = args[14]

	cdef long st_bu = 0
	cdef long st_bi = nU
	cdef long st_W = st_bi + nI
	cdef long st_C = st_W + nI**2

	cdef double pred_err = main_c_loop(
				&xlong[st_W], &xlong[st_C], &xlong[st_bu], &xlong[st_bi],
				&ix_u[0],
				&Rc[0], &Rd[0],
				&ix_u[0], &ix_i[0],
				&n_rated_by_user[0], &st_ix_user[0],
				&xlong[st_W], &xlong[st_C], &xlong[st_bu], &xlong[st_bi],
				alpha, lam1, lam2, lam3, lam4,
				nR, nI,
				0.0, 0, 0,
				nthreads
			)

	pred_err += lam1 * (np.linalg.norm(xlong[st_bu:st_bi])**2)\
				+ lam2 * (np.linalg.norm(xlong[st_bi:st_W])**2)\
				+ lam3 * (np.linalg.norm(xlong[st_W:st_C])**2)\
				+ lam4 * (np.linalg.norm(xlong[st_C:])**2)
	return pred_err/2

def calc_gradient(np.ndarray[double, ndim=1] xlong, *args):
	cdef np.ndarray[double, ndim=1] Rc = args[0]
	cdef np.ndarray[double, ndim=1] Rd = args[1]
	cdef np.ndarray[long, ndim=1] ix_u = args[2]
	cdef np.ndarray[long, ndim=1] ix_i = args[3]
	cdef np.ndarray[long, ndim=1] n_rated_by_user = args[4]
	cdef np.ndarray[long, ndim=1] st_ix_user = args[5]
	cdef long nR = args[6]
	cdef long nU = args[7]
	cdef long nI = args[8]
	cdef double alpha = args[9]
	cdef double lam1 = args[10]
	cdef double lam2 = args[11]
	cdef double lam3 = args[12]
	cdef double lam4 = args[13]
	cdef long nthreads = args[14]

	cdef long st_bu = 0
	cdef long st_bi = nU
	cdef long st_W = st_bi + nI
	cdef long st_C = st_W + nI**2

	cdef np.ndarray[double, ndim=1] grad = np.zeros(xlong.shape[0], dtype='float64')
	main_c_loop(
				&grad[st_W], &grad[st_C], &grad[st_bu], &grad[st_bi],
				&ix_u[0],
				&Rc[0], &Rd[0],
				&ix_u[0], &ix_i[0],
				&n_rated_by_user[0], &st_ix_user[0],
				&xlong[st_W], &xlong[st_C], &xlong[st_bu], &xlong[st_bi],
				alpha, lam1, lam2, lam3, lam4,
				nR, nI,
				-1.0, 1, 0,
				nthreads
			)
	return grad

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def optimize_sgd(
			np.ndarray[double, ndim=1] Rc,
			np.ndarray[double, ndim=1] Rd,
			np.ndarray[long, ndim=1] ix_u,
			np.ndarray[long, ndim=1] ix_i,
			np.ndarray[long, ndim=1] n_rated_by_user,
			np.ndarray[long, ndim=1] st_ix_user,
			long nR, long nU, long nI,
			np.ndarray[double, ndim=1] W,
			np.ndarray[double, ndim=1] C,
			np.ndarray[double, ndim=1] bu,
			np.ndarray[double, ndim=1] bi,
			double alpha, double lam1, double lam2,
			double lam3, double lam4, double initial_step_size,
			long maxiter, long decrease_step_size, long early_stop,
			long verbose, long random_seed, long use_seed):
	
	cdef long it
	cdef double err_iter
	cdef double err_prev = 10**12
	cdef np.ndarray[long, ndim=1] shuffled_ix = np.arange(nR)
	cdef double step_size = initial_step_size

	for it in range(maxiter):
		if use_seed != 0:
			np.random.seed(random_seed)
		np.random.shuffle(shuffled_ix)
		err_iter = main_c_loop_single(
				&W[0], &C[0], &bu[0], &bi[0],
				&shuffled_ix[0],
				&Rc[0], &Rd[0],
				&ix_u[0], &ix_i[0],
				&n_rated_by_user[0], &st_ix_user[0],
				&W[0], &C[0], &bu[0], &bi[0],
				alpha, lam1, lam2, lam3, lam4,
				nR, nI,
				step_size, 1, 1
			)
		if (verbose!=0):
			print "epoch " + str(it + 1) + " - error: " + str(err_iter)

		if (decrease_step_size!=0):
			step_size = initial_step_size * 0.98**(it + 1)

		if (early_stop!=0):
			if (err_prev - err_iter) <= 10:
				if it < 3:
					warnings.warn("SGD: Error did not decrease - try decreasing the step size. Aborted procedure.")
				break
		if err_iter > err_prev:
			step_size = step_size/2
		err_prev = err_iter

	return None
