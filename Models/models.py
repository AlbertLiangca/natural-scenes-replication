from funcs import *
import jax
import jax.numpy as jnp

class fixed_h_MaxEnt:
    def __init__(
        self,
        X0_init = 'random',
        a0 = 1,
        del_p = np.float32(1.05),
        del_n = np.sqrt(2,dtype=np.float32)
    ):
        self.a0 = a0
        self.X0_init = 'random'
        self.del_p = del_p
        self.del_n = del_n
        self.X = None

    def fit(self,sigma):
        B,N = sigma.shape
        D = int(N+N*(N-1)/2)
        if self.X0_init == 'random':
            X0 = random.rand(D)
        else:
            print('not implemented yet!')
            return None


        combs_obs = all_combs_obs(N)
        Z0 = Z(X0,combs_obs)
        Q0 = Q(X0,combs_obs,Z0)
        P_ = P_bar(sigma)
        X_ = susc_bar(sigma)
        inv_X_ = jnp.linalg.pinv(X_,hermitian=True)
        e0 = epsilon(P_,inv_X_,Q0,B)

        et = e0
        Qt = Q0
        Xt = X0
        at = self.a0
        Zt = Z0

        while et >= 1:
            Mt = jnp.ceil(jnp.minimum(B/et**2,B)).astype(jnp.int32)
            del_Xt = at*inv_X_@(P_ - Qt)
            Xt_1 = Xt + del_Xt
            Qt_1 = QMC(sigma,Mt)
            et_1 = epsilon(P_,inv_X_,Qt_1,B)
            if et_1 < et:
                Qt = Qt_1
                Xt = Xt_1
                et = et_1
                at_1 = at*self.del_p
                at = at_1
            else:
                at_1 = at/self.del_n
                at = at_1
        
        Xf = Xt
        Zf = Zt
        ef = et
        self.X = Xf
        self.Z = Zf

        return Xf,ef
    
    def predict(self,sigmat):
        if self.X == None:
            print('Hasn\'t been fit yet!')
            return None
        return pt(sigmat,X=self.X,Z = self.Z)     