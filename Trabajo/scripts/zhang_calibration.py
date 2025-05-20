def zhang_calibration(object_points_list, image_points_list):
    A = []
    B = []
    
    for i in range(len(image_points_list)):
        image_points = image_points_list[i]
        object_points = object_points_list[i]
        
        for j in range(len(image_points)):
            X, Y, Z = object_points[j]
            x, y = image_points[j].ravel()
            
            A.append([X, Y, Z, 1, 0, 0, 0, 0, -x * X, -x * Y, -x * Z, -x])
            A.append([0, 0, 0, 0, X, Y, Z, 1, -y * X, -y * Y, -y * Z, -y])
            B.append([x])
            B.append([y])

    A = np.array(A)
    B = np.array(B) 

    P = np.linalg.lstsq(A, B, rcond=None)[0]  
 
    P = P.reshape(3, 4)
    P = P / P[2, 3]
    return P
