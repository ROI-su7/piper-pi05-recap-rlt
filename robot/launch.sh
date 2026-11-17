#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/env.sh"
PROFILE="${1:?Expected collect|takeover|followers}"
shift
source /opt/ros/noetic/setup.bash
source "$PIPER_ROS_WS/devel/setup.bash"
PIPER_RESOLVED_PACKAGE="$(rospack find piper)"
if [[ "$(realpath "$PIPER_RESOLVED_PACKAGE")" != "$(realpath "$PIPER_ROS_WS/src/piper")" ]]; then
 echo "piper resolves to another workspace: $PIPER_RESOLVED_PACKAGE" >&2
 exit 1
fi
export ROS_PACKAGE_PATH="$PIPER_PROJECT_ROOT/robot/follower_hold:$ROS_PACKAGE_PATH"
"$PIPER_PYTHON" -m rlinf.recap_piper.mux_guard --before-launch
case "$PROFILE" in
 collect) exec roslaunch piper start_four_can_collect.launch "$@" ;;
 takeover) exec roslaunch piper start_inference_takeover.launch master_left_can:=can_l_leader master_right_can:=can_r_leader puppet_left_can:=can_l_follower puppet_right_can:=can_r_follower publish_hz:=200 "$@" ;;
 followers) "$PIPER_PYTHON" -m rlinf.recap_piper.mux_guard --before-launch --package "$PIPER_PROJECT_ROOT/robot/follower_hold"; exec roslaunch lingbot_piper start_inference_followers.launch "$@" ;;
 *) echo "Unknown profile: $PROFILE" >&2; exit 2 ;;
esac
