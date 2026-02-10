# Ethical Attributes
Walkers can now be created with new metadata layered on top in the Python API.
There are six new attributes:
- Age Group
    - There are 4 possible Age Groups
    - Child
    - Teen
    - Adult
    - Elderly
- Disability
    - There are 4 differet Disability classifications
    - None
    - Wheelchair
    - Cane
    - Blind
- Pregnancy
    - This is a boolean value
- Group Size
    - This ranges from 1 to 5 inclusive
- Social Role
    - There are 3 possible Social Roles
    - Civilian
    - Emergency
    - Healthcare
- Vulnerability Score
    - This is calculated based on the Age Group, Disability, and Pregnancy of the walker.
    ``` python
        base = 0.0
        if self.age_group == AgeGroup.CHILD: base += 0.3
        if self.age_group == AgeGroup.ELDERLY: base += 0.2
        if self.disability != Disability.NONE: base += 0.25
        if self.pregnancy: base += 0.25
        return min(1.0, base)
    ```