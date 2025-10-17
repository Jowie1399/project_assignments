#Assigning the smartphone class
class smartphone:
  def __int__(self,brand,model, battery_level):
    self.brand=brand
    self.model=model
    self.__battery_level=battery_level#encapsulating battery level,that is hiding it from overwriting 

  # creating a call method on the smartphone class
  def call(self,contact):
    print(f"Calling {contact} using {self.model}")

#method for charging 
def charge(self,charge):
  self.__battery_level=min(100,self.__battery_level+amount)
  print((f"🔋 {self.model} charged. Battery level: {self.__battery_level}%"))

#method for battery level
def battery(self):
  print(f" battery level is at {self.__battery_level}")

#inherited class
class smartwatch(smartphone):
  def __init__(self,brand,model,battery_level,steps=0):
    super().__init__(brand,model,battery_level)
    self.steps=steps
    
    #polymorphism at work
  def call(self,contact):
    print(f"calling {contact} by {self.model} smartwatch..")

  def track_steps(self,steps_walked):
    self.steps=steps_walked
    print(f"Steps tracked: {self.steps}")

# --- Test Section ---
phone = Smartphone("Samsung", "Galaxy S23", 70)
watch = Smartwatch("Apple", "Watch Series 9", 80)

phone.call("Alice")
watch.call("Bob")

phone.charge(20)
watch.track_steps(3000)

print(f"📱 {phone.model} battery: {phone.get_battery()}%")
print(f"⌚ {watch.model} battery: {watch.get_battery()}%")
  
#Activity 2
class vehicle ():
  def move(self):
    print("Vehicles move im one way")

class car(vehicle):
  def move(self):
  print("A car moves om the road")

class boat(vehicle):
  def move(self):
    print("A boat sails in the ocean")

class plane(vehicle):
  def move(self)
  print("A planr flies in the sky") 

# Polymorphism demonstration
vehicles = [Car(), Plane(), Boat()]

print("\n--- Polymorphism in Action ---")
for v in vehicles:
    v.move()


  
