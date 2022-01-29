import re


# CONSTS / IDs
MONITOR_NUMBER = "MONITOR_NUMBER"


# Id should be string
def set_config_by_id(id, new_value):
    with open("config.cfg", 'r+') as f:
        text = f.read()
        text = re.sub(re.compile(str(id) + ".*"), str(id) + "=" + str(new_value), text)
        f.seek(0)
        f.write(text)
        f.truncate()


def get_config_value_by_id(id):
    with open("config.cfg", 'r+') as f:
        text = f.read()
        f.close()
    line = re.compile(str(id) + ".*").findall(text)[0]
    return line.split("=")[1]


if __name__ == "__main__":
    print(get_config_value_by_id(MONITOR_NUMBER))
    # set_config_by_id(MONITOR_NUMBER, "1.005")
