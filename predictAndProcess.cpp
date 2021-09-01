cv::Mat predictAndProcess(cv::Mat inputs)
{
    cv::Mat inputBlob = cv::dnn::blobFromImage(inputs, 1.0, cv::Size(256, 256), false, false);

    // 读入四个网络得到输出mask -> (Batch, Channel, Height, Width)
    vegetation_net.setInput(inputBlob);
    cv::Mat mask1 = vegetation_net.forward();

    bldg_net.setInput(inputBlob);
    cv::Mat mask2 = bldg_net.forward();

    water_net.setInput(inputBlob);
    cv::Mat mask3 = water_net.forward();

    road_net.setInput(inputBlob);
    cv::Mat mask4 = road_net.forward();


    const int rows = mask1.size[2];
    const int cols = mask1.size[3];

    cv::Mat value1(rows, cols, CV_32FC1, mask1.data);
    cv::Mat value2(rows, cols, CV_32FC1, mask2.data);
    cv::Mat value3(rows, cols, CV_32FC1, mask3.data);
    cv::Mat value4(rows, cols, CV_32FC1, mask4.data);
    cv::Mat segm(rows, cols, CV_8UC3);

    for (int row = 0; row < rows; row++)
    {
        float *ptrSource1 = value1.ptr<float>(row);   // vegetation
        float *ptrSource2 = value2.ptr<float>(row);   // bldg
        float *ptrSource3 = value3.ptr<float>(row);   // water
        float *ptrSource4 = value4.ptr<float>(row);   // road

        cv::Vec3b *ptrSegm = segm.ptr<cv::Vec3b>(row);
        for (int col = 0; col < cols; col++)
        {
            // 根据阈值决定所属类别
            ptrSource1[col] = ptrSource1[col] >= 0.1f? 1.0f : 0;
            ptrSource2[col] = ptrSource2[col] >= 0.3f? 1.0f : 0;
            ptrSource3[col] = ptrSource3[col] >= 0.5f? 1.0f : 0;
            ptrSource4[col] = ptrSource4[col] >= 0.45f? 1.0f : 0;

            if (ptrSource1[col] == 1.0f)   ptrSegm[col] = colors[1];

            if (ptrSource2[col] == 1.0f)    ptrSegm[col] = colors[2];

            if (ptrSource3[col] == 1.0f)    ptrSegm[col] = colors[3];

            if (ptrSource4[col] == 1.0f)    ptrSegm[col] = colors[4];
        }
    }

    return seg;
}
