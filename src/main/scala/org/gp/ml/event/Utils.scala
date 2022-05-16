package org.gp.ml.event

import org.tensorflow.framework.Summary
import org.tensorflow.util.Event

import java.time.Instant

object Utils {
  def myEvent(name: String, v: Float, step: Int): Event = {
    val x = Summary.Value.newBuilder()

    x.setTag(name)
    x.setSimpleValue(v)

    val value = x.build()
    val summary = Summary.newBuilder().addValue(value).build()
    Event.newBuilder()

    Event.newBuilder().setSummary(summary).setStep(step).setWallTime(Instant.now.getEpochSecond).build()
  }
}
