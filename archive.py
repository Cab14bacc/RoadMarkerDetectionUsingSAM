# classify splines code where spline_data ordering is not properly ordered (in real world) from previous steps
def classify_splines(spline_data, groups, image, white_color = [225, 225, 225], yellow_color = [112, 206, 244], debug_path = None):


    if (isinstance(image, np.ndarray)):
        source_image = image.copy()
    elif os.path.isfile(image):
        source_image = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        source_image = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)

    height, width = source_image.shape[:2]
    max_extension_length = int(np.sqrt(height * height + width * width)) 
    
    if debug_path is not None:
        combined_debug_image = np.zeros_like(source_image)

    spline_types = []

    classified_spline_data = []



    for group_idx, group in enumerate(groups):
        # print(f"current group: {group_idx}")

        component_spline_data = groups_to_spline_data([group], spline_data)
        local_groups = group_by_direction_and_extension(component_spline_data, 8,
                                                        max_projection=max_extension_length)
        
        component_spline_arc_lengths = []

        for spline_datum in spline_data:
            spline_pts = spline_datum["spline_pts"]
            component_spline_arc_lengths.append(compute_spline_length(spline_pts))

        # select the 2 groups with the largest arc length if more than 2 groups
        if len(local_groups) > 2:
            group_arc_lengths = []

            for local_group in local_groups:
                current_group_arc_length = 0
                for local_idx in local_group:
                    current_group_arc_length += component_spline_arc_lengths[local_idx]

                group_arc_lengths.append(current_group_arc_length)

            sorted_indices = np.argsort(group_arc_lengths)[::-1]
            local_groups = local_groups[sorted_indices][:2]


            # remove those unselect from previous step
            classified_component_spline_data = []
            classified_local_groups = []
            classified_component_spline_arc_lengths = []

            for local_group in local_groups:
                classified_local_groups.append([])
                for local_idx in local_group:
                    classified_local_groups.append(len(classified_component_spline_data))
                    classified_component_spline_arc_lengths.append(component_spline_arc_lengths[local_idx])
                    classified_component_spline_data.append(component_spline_data[local_idx])
            
            component_spline_data = classified_component_spline_data
            local_groups = classified_local_groups
            component_spline_arc_lengths = classified_component_spline_arc_lengths


        # sample points from groups to obtain color
        component_colors = []

        for local_group in local_groups:
            current_group_colors = []
            for local_idx in local_group:
                spline_pts = component_spline_data[local_idx]["spline_pts"]
                for spline_pt in spline_pts:
                    x, y = spline_pt
                    color = source_image[y, x]
                    current_group_colors.append(color)
                
            dist_to_white = np.linalg.norm(np.array(current_group_colors) - np.array(white_color), axis=1) # (num of spline pts in component, 3)
            dist_to_white = np.mean(dist_to_white)

            dist_to_yellow = np.linalg.norm(np.array(current_group_colors) - np.array(yellow_color), axis=1)
            dist_to_yellow = np.mean(dist_to_yellow)

            component_color = "yellow" if dist_to_white > dist_to_yellow else "white"
            component_colors.append(component_color) 


        # determine "dashed" or "solid" grouped splines
        # compute the order of the splines within a local group to calculate the average gap
        # we achieve this by computing the nearest endpoint from this endpoint (excluding itself and endpoint from same spline) 
        component_line_styles = []


        for local_group in local_groups:
            print(local_group)
            if len(local_group) == 1:
                component_line_styles.append("solid")
                continue



            
            spline_endpoints = []

            for local_idx in local_group:
                start = component_spline_data[local_idx]["start"]
                end = component_spline_data[local_idx]["end"]
                spline_endpoints.append(start)
                spline_endpoints.append(end)
            
            spline_endpoints = np.array(spline_endpoints)

            # a NxN array, where [i, j] contains the distance of i endpoint to j endpoint 
            spline_endpoint_dists = np.linalg.norm(spline_endpoints[None, ...] - spline_endpoints[:, None, :], axis=-1)

            
            
            # excluding itself and endpoint from same spline
            for i in range(0, len(spline_endpoints), 2):
                start_idx = i
                end_idx = i + 1

                spline_endpoint_dists[start_idx][start_idx] = np.inf
                spline_endpoint_dists[end_idx][end_idx] = np.inf
                spline_endpoint_dists[start_idx][end_idx] = np.inf
                spline_endpoint_dists[end_idx][start_idx] = np.inf
            
            
            def dfs(current_spline_pos_within_ordered_list, consider, consider_dir):

                current_spline_idx = ordered_spline_indices[current_spline_pos_within_ordered_list]

                spline_pos_within_local_group = local_group.index(current_spline_idx)
                visited_splines[spline_pos_within_local_group] = True

                nearest_endpoint_pos_within_local_group_to_start = min_endpoint_dists_indices[2 * spline_pos_within_local_group]
                nearest_endpoint_pos_within_local_group_to_end = min_endpoint_dists_indices[2 * spline_pos_within_local_group + 1]

                nearest_spline_pos_within_local_group_to_start = int(nearest_endpoint_pos_within_local_group_to_start // 2)
                nearest_spline_to_start_consider = "start" if nearest_endpoint_pos_within_local_group_to_start / 2 > nearest_spline_pos_within_local_group_to_start else "end"

                if "start" in consider and not visited_splines[nearest_spline_pos_within_local_group_to_start]:
                    visited_splines[nearest_spline_pos_within_local_group_to_start] = True
                    ordered_spline_indices.insert(current_spline_pos_within_ordered_list + consider_dir[consider.index("start")], local_group[nearest_spline_pos_within_local_group_to_start])
                    print(f"inserting {local_group[nearest_spline_pos_within_local_group_to_start]}, currently {ordered_spline_indices}")

                    inserted_pos_within_ordered_list = dfs(current_spline_pos_within_ordered_list, [nearest_spline_to_start_consider], [consider_dir[consider.index("start")]])

                    if consider_dir[consider.index("start")] == 0:
                        current_spline_pos_within_ordered_list = inserted_pos_within_ordered_list + 1





                nearest_spline_pos_within_local_group_to_end = int(nearest_endpoint_pos_within_local_group_to_end // 2)
                nearest_spline_to_end_consider = "start" if nearest_endpoint_pos_within_local_group_to_end / 2 > nearest_spline_pos_within_local_group_to_end else "end"

                if "end" in consider and not visited_splines[nearest_spline_pos_within_local_group_to_end]:
                    visited_splines[nearest_spline_pos_within_local_group_to_end] = True
                    ordered_spline_indices.insert(current_spline_pos_within_ordered_list + consider_dir[consider.index("end")], local_group[nearest_spline_pos_within_local_group_to_end])
                    print(f"inserting {local_group[nearest_spline_pos_within_local_group_to_end]}, currently {ordered_spline_indices}")

                    inserted_pos_within_ordered_list = dfs(current_spline_pos_within_ordered_list + 1, [nearest_spline_to_end_consider], [consider_dir[consider.index("end")]])
                    
                    if consider_dir[consider.index("end")] == 0:
                        current_spline_pos_within_ordered_list = inserted_pos_within_ordered_list + 1


                return current_spline_pos_within_ordered_list
                

            # the index (within list) of the nearest endpoint
            min_endpoint_dists_indices = np.argmin(spline_endpoint_dists, axis=0) #Nx1

            ordered_spline_indices = [local_group[0]]
            print(f"staring ordering with spline {local_group[0]}")
            visited_splines = [False] * len(local_group)
            current_spline_pos_within_ordered_list = 0

            dfs(current_spline_pos_within_ordered_list, ["start", "end"], [0, 1])

            print(ordered_spline_indices)


        if debug_path is not None:
            colors = [255 * np.array(hsv_to_rgb(i * (1 / (len(local_groups))), 1, 1)) for i in range(len(local_groups))]
            debug_image = draw_grouped_splines(component_spline_data, local_groups, np.zeros_like(source_image),colors)
            for i, spline_datum in enumerate(component_spline_data):
                debug_pt = spline_datum["start"]
                cv2.putText(debug_image, f"{i}", np.array([debug_pt[0], min(debug_pt[1] + 100, height)]).astype(np.int32), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 3, cv2.LINE_AA)


            result, encoded_img = cv2.imencode('.png', debug_image)
            encoded_img.tofile(os.path.join(debug_path, 'debug_classify', f'classify_spline_debug_{group_idx}.png'))        
            combined_debug_image = np.where(np.any(debug_image, -1)[..., None], debug_image, combined_debug_image)
            # debug_image = cv2.resize(debug_image, (0, 0), fx=0.1, fy=0.1)
            # cv2.imshow('spline_debug', debug_image)
            # cv2.waitKey(0)


        spline_types.append(component_color)

    if debug_path is not None:
        result, encoded_img = cv2.imencode('.png', combined_debug_image)
        encoded_img.tofile(os.path.join(debug_path, f'combined_classify_spline_debug.png'))

    return spline_types     
   