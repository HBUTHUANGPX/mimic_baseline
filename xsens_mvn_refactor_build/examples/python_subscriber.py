#!/usr/bin/env python3
import argparse
import sys

import zmq


def parse_args():
    parser = argparse.ArgumentParser(description="Subscribe xsens link_states protobuf over ZMQ")
    parser.add_argument(
        "--connect",
        default="tcp://127.0.0.1:5555",
        help="ZMQ publisher address, default: tcp://127.0.0.1:5555",
    )
    parser.add_argument(
        "--topic",
        default="xsens.link_states.v1",
        help="Subscription topic, default: xsens.link_states.v1",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=2,
        help="How many protobuf messages to receive before exit",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=5000,
        help="Receive timeout in milliseconds",
    )
    parser.add_argument(
        "--proto-module",
        default="link_states_pb2",
        help="Generated Python protobuf module name",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        proto_module = __import__(args.proto_module)
    except ImportError as exc:
        print(
            f"failed_to_import_proto_module={args.proto_module} error={exc}",
            file=sys.stderr,
        )
        return 1

    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, args.topic)
    subscriber.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    subscriber.connect(args.connect)

    print(f"subscriber_connect={args.connect}")
    print(f"subscriber_topic={args.topic}")
    print(f"subscriber_count={args.count}")

    for _ in range(args.count):
      topic = subscriber.recv_string()
      payload = subscriber.recv()

      message = proto_module.LinkStateArray()
      message.ParseFromString(payload)

      print(f"topic={topic}")
      print(f"schema_version={message.header.schema_version}")
      print(f"frame_id={message.header.frame_id}")
      print(f"states_size={len(message.states)}")
      for index, state in enumerate(message.states):
          print(
              f"state[{index}].name={state.name} "
              f"position={state.pose.position.x},{state.pose.position.y},{state.pose.position.z}"
          )

    subscriber.close(0)
    context.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
