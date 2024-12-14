'use client';

import * as React from "react"

const Card = React.forwardRef(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className="rounded-lg border bg-white shadow-sm p-4"
    {...props}
  />
))
Card.displayName = "Card"

export { Card }