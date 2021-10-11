
def truncate_sdf(sdf, truncation_val):
    sdf[sdf > truncation_val] = truncation_val
    sdf[sdf < -truncation_val] = -truncation_val
    return sdf
