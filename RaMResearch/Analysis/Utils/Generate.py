######################################################
###### Class to automate generation of assets ########
######################################################


# Generate all ring images
def generate_all_rotations(r_small, r_large, anglerange=(1, 90)):
    import RaMResearch.Data.RingV2 as rv2

    point_cloud = rv2.RingPointCloud(r_large, r_small)
    ring_im_obj = point_cloud.get_image(outline=False, morph_operations=True, angle=0)
    for i in range(anglerange[0], anglerange[1]):
        rot_ring_im_obj = rv2.create_rotated_image(ring_im_obj, i)
        rot_ring_im_obj.save()
        print("Created Ring Image with Rotatation:\t" + str(i))


generate_all_rotations(int(55 / 2), 200, anglerange=(90, 180))
