import pybase

print("PyBase Shell")
print("v1.0.0")
while True:
    text = input("PyBase >>> ")
    if text == "exit":
        break
    elif text == "about":
        print("This is PyBase: A BASIC-like language written completely in Python!")
        continue
    if text.strip() == "": continue
    result, error = pybase.run("<stdin>", text)

    if error: print(error.as_string())
    elif result: print(result)