"""Invert the frozen monotone MPM angle-volume samples."""

def linear_proposal(points, target):
    points = sorted(points)
    if len(points)<2 or any(b[0]<=a[0] or b[1]<=a[1] for a,b in zip(points[:-1],points[1:])):
        raise ValueError('Simulation curve must increase; preserve and review any reversal')
    if target<=points[0][1]:
        a,b = points[:2]
    elif target>=points[-1][1]:
        a,b = points[-2:]
    else:
        a,b = next((a,b) for a,b in zip(points[:-1],points[1:]) if a[1]<=target<=b[1])
    angle = round(a[0]+(target-a[1])*(b[0]-a[0])/(b[1]-a[1]),2)
    if not 45.<=angle<=68.:
        raise ValueError('Proposed angle outside the frozen range')
    return angle
