
#----------------------------------------------------------------------------------------------------
# Please check the licenses of the respective works utilized here before using this script.
#----------------------------------------------------------------------------------------------------

docker build --network host -t rpx .
docker tag rpx anonymous/rpx:latest
