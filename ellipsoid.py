import numpy as np

def get_cov_ellipsoid(cov, mu=np.zeros((3)), nstd=3):
    """
    Return the 3d points representing the covariance matrix
    cov centred at mu and scaled by the factor nstd.

    Plot on your favourite 3d axis. 
    Example 1:  ax.plot_wireframe(X,Y,Z,alpha=0.1)
    Example 2:  ax.plot_surface(X,Y,Z,alpha=0.1)
    """
    assert cov.shape==(3,3)

    # Find and sort eigenvalues to correspond to the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.sum(cov,axis=0).argsort()
    eigvals_temp = eigvals[idx]
    idx = eigvals_temp.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]

    # Set of all spherical angles to draw our ellipsoid
    n_points = 100
    theta = np.linspace(0, 2*np.pi, n_points)
    phi = np.linspace(0, np.pi, n_points)

    # Width, height and depth of ellipsoid
    rx, ry, rz = nstd * np.sqrt(eigvals)

    # Get the xyz points for plotting
    # Cartesian coordinates that correspond to the spherical angles:
    X = rx * np.outer(np.cos(theta), np.sin(phi))
    Y = ry * np.outer(np.sin(theta), np.sin(phi))
    Z = rz * np.outer(np.ones_like(theta), np.cos(phi))

    # Rotate ellipsoid for off axis alignment
    old_shape = X.shape
    # Flatten to vectorise rotation
    X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()
    X,Y,Z = np.matmul(eigvecs, np.array([X,Y,Z]))
    X,Y,Z = X.reshape(old_shape), Y.reshape(old_shape), Z.reshape(old_shape)
   
    # Add in offsets for the mean
    X = X + mu[0]
    Y = Y + mu[1]
    Z = Z + mu[2]
    
    return X,Y,Z



#########################################################
## Testing some data
#########################################################
if __name__=='__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Setup the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.xlabel('x')
    plt.ylabel('y')
    # ax.set_aspect(1.0)
    # ax.set_xlim3d(-3,10)
    # ax.set_ylim3d(-3,10)
    # ax.set_zlim3d(-3,10)
    # Generate random 3d data points for 2 different offsets

    #########################################################
    ## Generate datasets s1 and s2
    #########################################################
    # s1
    mu1 = np.random.random((3)) * 5
    cov1 = np.array([            # Using an off-axis alignment for s1
        [2.5, 0.75, 0.175],
        [0.75, 0.70, 0.135],
        [0.175, 01.35, 0.43]
    ])
    s1 = np.random.multivariate_normal(mu1, cov1, (200))
    ax.scatter(s1[:,0],s1[:,1],s1[:,2], c='r')

    # s2
    mu2 = np.random.random((3)) * 5 + 4
    cov2 = np.diag((1,3,5))
    s2 = np.random.multivariate_normal(mu2, cov2, (200))
    ax.scatter(s2[:,0],s2[:,1],s2[:,2], c='b')

    #########################################################
    ## Process data and plot ellipsoid
    #########################################################
    nstd = 2    # 95% confidence interval
    # s1
    mu1_ = np.mean(s1, axis=0)
    cov1_ = np.cov(s1.T)
    X1,Y1,Z1 = get_cov_ellipsoid(cov1_, mu1_, nstd)
    ax.plot_wireframe(X1,Y1,Z1, color='r', alpha=0.1)

    # s2
    mu2_ = np.mean(s2, axis=0)
    cov2_ = np.cov(s2.T)
    X2,Y2,Z2 = get_cov_ellipsoid(cov2_, mu2_, nstd)
    ax.plot_wireframe(X2,Y2,Z2, color='b', alpha=0.1)

    plt.show()